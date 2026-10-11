"""MedEnergy: offline beginner demo, using only the Python standard library.

Run python medenergy.py. Inputs and parameters are synthetic, not clinical
specifications. One day-ahead forecast is fitted on the first 24 days and
evaluated on the final 6 days. Only non-clinical batch jobs may be moved.
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DT_HOURS = 0.25
SLOTS_PER_DAY = 96


def load_config() -> dict:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    if not 2 <= config["training_days"] < config["days"] <= 365:
        raise ValueError("Require 2 <= training_days < days <= 365")
    if config["sterilizer_duration_minutes"] not in (15, 30, 45, 60):
        raise ValueError("sterilizer_duration_minutes must be 15, 30, 45 or 60")
    if not 0 <= config["max_shift_hours"] <= 6:
        raise ValueError("max_shift_hours must be between 0 and 6")
    if config["pv_capacity_kw"] < 0 or config["sterilizer_power_kw"] <= 0:
        raise ValueError("PV capacity must be non-negative; task power must be positive")
    if config["carbon_weight_yuan_per_kg"] < 0:
        raise ValueError("carbon weight must be non-negative")
    return config


def make_tasks(config: dict, workday: bool) -> list[dict]:
    """Two teaching-example sterilization batches with disjoint time windows.

    The second window starts after the first deadline, so tasks cannot overlap.
    These windows do not describe a real hospital's clinical workflow.
    """
    if not workday:
        return []
    common = {
        "duration": int(config["sterilizer_duration_minutes"] // 15),
        "power_kw": float(config["sterilizer_power_kw"]),
    }
    return [
        {**common, "name": "模拟灭菌批次 A", "release": 32, "deadline": 56, "baseline_start": 36},
        {**common, "name": "模拟灭菌批次 B", "release": 56, "deadline": 84, "baseline_start": 72},
    ]


def baseline_task_load(tasks: list[dict]) -> list[float]:
    load = [0.0] * SLOTS_PER_DAY
    for task in tasks:
        for slot in range(task["baseline_start"], task["baseline_start"] + task["duration"]):
            load[slot] += task["power_kw"]
    return load


def generate_data(config: dict) -> tuple[list[dict], list[list[dict]]]:
    """Generate actual measurements and separately available forecast inputs."""
    rng = random.Random(config["seed"])
    start = datetime(2026, 9, 1)
    rows, tasks_by_day = [], []
    for day in range(config["days"]):
        date = start + timedelta(days=day)
        workday = date.weekday() < 5
        tasks = make_tasks(config, workday)
        tasks_by_day.append(tasks)
        flexible_load = baseline_task_load(tasks)
        weather_trend = 3.5 * math.sin(day * 2 * math.pi / 9)
        clouds = 0.65 + 0.25 * rng.random()
        for slot in range(SLOTS_PER_DAY):
            ts = date + timedelta(minutes=15 * slot)
            hour = slot / 4
            daytime = max(0.0, math.sin((hour - 7) * math.pi / 12)) if 7 <= hour <= 19 else 0
            # A simulated forecast, known before the day starts.
            forecast_temp = 24 + weather_trend + 5 * math.sin((hour - 8) * math.pi / 12)
            planned_occupancy = 0.30 + (0.55 if workday else 0.18) * daytime
            # Measurements differ from the forecast; these are never model inputs.
            actual_temp = forecast_temp + rng.gauss(0, 0.9)
            actual_occupancy = max(0.1, min(1.0, planned_occupancy + rng.gauss(0, 0.035)))
            # Imaging is a fixed appointment schedule, never shifted by this demo.
            imaging_kw = 5.0
            if workday and 8 <= hour < 18:
                imaging_kw += 32.0 if slot % 8 in (0, 1, 2, 3) else 8.0
            critical_kw = 72 + 6 * actual_occupancy
            hvac_kw = max(0, 48 + 5.5 * max(actual_temp - 22, 0) + 34 * actual_occupancy + rng.gauss(0, 3))
            lighting_kw = max(0, 7 + 14 * actual_occupancy + rng.gauss(0, 0.7))
            background_kw = critical_kw + imaging_kw + hvac_kw + lighting_kw
            solar_shape = max(0, math.sin((hour - 6) * math.pi / 12)) if 6 <= hour <= 18 else 0
            pv_forecast = config["pv_capacity_kw"] * solar_shape * clouds
            pv_actual = max(0, pv_forecast * (1 + rng.gauss(0, 0.07))) if pv_forecast else 0
            # All price / carbon values are invented scenario inputs.
            tariff = 0.42 if hour < 7 or hour >= 23 else (1.05 if 8 <= hour < 11 or 17 <= hour < 21 else 0.72)
            carbon = 0.43 if hour < 7 or hour >= 23 else (0.64 if 17 <= hour < 21 else (0.36 if 11 <= hour < 16 else 0.54))
            rows.append({
                "timestamp": ts.isoformat(timespec="minutes"),
                "day_index": day,
                "hour": hour,
                "weekend": int(not workday),
                "forecast_temperature_c": forecast_temp,
                "measured_temperature_c": actual_temp,
                "planned_occupancy_ratio": planned_occupancy,
                "measured_occupancy_ratio": actual_occupancy,
                "imaging_fixed_kw": imaging_kw,
                "critical_kw": critical_kw,
                "hvac_kw": hvac_kw,
                "lighting_kw": lighting_kw,
                "actual_background_kw": background_kw,
                "baseline_task_kw": flexible_load[slot],
                "actual_total_kw": background_kw + flexible_load[slot],
                "pv_forecast_kw": pv_forecast,
                "pv_actual_kw": pv_actual,
                "tariff_yuan_per_kwh": tariff,
                "carbon_kg_per_kwh": carbon,
            })
    return rows, tasks_by_day


def features(row: dict) -> list[float]:
    """Only information available before the forecast day starts."""
    return [
        1.0,
        max(row["forecast_temperature_c"] - 22, 0) / 10,
        row["planned_occupancy_ratio"],
        row["imaging_fixed_kw"] / 40,
        math.sin(row["hour"] * math.pi / 12),
        math.cos(row["hour"] * math.pi / 12),
        float(row["weekend"]),
    ]


def fit_linear_regression(train_rows: list[dict]) -> list[float]:
    """Fit a small ridge regression using normal equations, no third-party libs."""
    size = len(features(train_rows[0]))
    matrix = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row in train_rows:
        x, y = features(row), row["actual_background_kw"]
        for i in range(size):
            rhs[i] += x[i] * y
            for j in range(size):
                matrix[i][j] += x[i] * x[j]
    for i in range(1, size):
        matrix[i][i] += 0.05
    for i in range(size):
        pivot = max(range(i, size), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        rhs[i], rhs[pivot] = rhs[pivot], rhs[i]
        divisor = matrix[i][i]
        if abs(divisor) < 1e-12:
            raise ValueError("Regression matrix is singular")
        for j in range(i, size):
            matrix[i][j] /= divisor
        rhs[i] /= divisor
        for k in range(size):
            if k == i:
                continue
            factor = matrix[k][i]
            for j in range(i, size):
                matrix[k][j] -= factor * matrix[i][j]
            rhs[k] -= factor * rhs[i]
    return rhs


def predict(row: dict, coefficients: list[float]) -> float:
    return max(0.0, sum(a * b for a, b in zip(features(row), coefficients)))


def errors(actual: list[float], estimated: list[float]) -> dict:
    residuals = [a - b for a, b in zip(actual, estimated)]
    return {
        "mae_kw": sum(abs(e) for e in residuals) / len(residuals),
        "rmse_kw": math.sqrt(sum(e * e for e in residuals) / len(residuals)),
    }


def optimize_day(day_rows: list[dict], tasks: list[dict], predicted_background: list[float], config: dict) -> tuple[list[float], list[int]]:
    """Enumerate every permitted continuous start; use forecast values to plan.

    Disjoint task windows make this separable. This is not a general hospital
    scheduling solver. It preserves each task and never reschedules imaging.
    """
    shift_slots = int(config["max_shift_hours"] * 4)
    alpha = config["carbon_weight_yuan_per_kg"]
    planned_load = [0.0] * SLOTS_PER_DAY
    starts = []
    for task in tasks:
        earliest = max(task["release"], task["baseline_start"] - shift_slots)
        latest = min(task["deadline"] - task["duration"], task["baseline_start"] + shift_slots)
        if earliest > latest:
            raise ValueError("Task has no feasible start")

        def score(start: int) -> tuple[float, int]:
            marginal = 0.0
            for slot in range(start, start + task["duration"]):
                row = day_rows[slot]
                net = predicted_background[slot] + planned_load[slot] - row["pv_forecast_kw"]
                extra_grid = max(net + task["power_kw"], 0) - max(net, 0)
                marginal += extra_grid * DT_HOURS * (row["tariff_yuan_per_kwh"] + alpha * row["carbon_kg_per_kwh"])
            # Prefer less movement if objective values tie.
            return (marginal, abs(start - task["baseline_start"]))

        chosen = min(range(earliest, latest + 1), key=score)
        starts.append(chosen)
        for slot in range(chosen, chosen + task["duration"]):
            planned_load[slot] += task["power_kw"]
    return planned_load, starts


def evaluate_day(day_rows: list[dict], task_load: list[float]) -> dict:
    """Evaluate on actual measurements, never on optimistic forecast values."""
    total = [row["actual_background_kw"] + load for row, load in zip(day_rows, task_load)]
    grid = [max(0, power - row["pv_actual_kw"]) for row, power in zip(day_rows, total)]
    used_pv = [min(row["pv_actual_kw"], power) for row, power in zip(day_rows, total)]
    return {
        "grid_kw": grid,
        "total_kw": total,
        "cost_yuan": sum(g * row["tariff_yuan_per_kwh"] * DT_HOURS for g, row in zip(grid, day_rows)),
        "carbon_kg": sum(g * row["carbon_kg_per_kwh"] * DT_HOURS for g, row in zip(grid, day_rows)),
        "peak_kw": max(grid),
        "total_energy_kwh": sum(total) * DT_HOURS,
        "grid_energy_kwh": sum(grid) * DT_HOURS,
        "used_pv_kwh": sum(used_pv) * DT_HOURS,
    }


def verify_plan(day_rows: list[dict], tasks: list[dict], load: list[float], starts: list[int], config: dict) -> None:
    """Check physical invariants needed for truthful comparison."""
    baseline = baseline_task_load(tasks)
    if not math.isclose(sum(baseline), sum(load), rel_tol=1e-10, abs_tol=1e-8):
        raise AssertionError("Flexible-task energy must be conserved")
    occupied = set()
    for task, start in zip(tasks, starts):
        if not task["release"] <= start or start + task["duration"] > task["deadline"]:
            raise AssertionError("Task release/deadline violated")
        if abs(start - task["baseline_start"]) > int(config["max_shift_hours"] * 4):
            raise AssertionError("Maximum shift violated")
        for slot in range(start, start + task["duration"]):
            if slot in occupied:
                raise AssertionError("One sterilizer cannot perform two tasks at once")
            occupied.add(slot)
            if load[slot] != task["power_kw"]:
                raise AssertionError("Task duration or power changed")
    before = evaluate_day(day_rows, baseline)
    after = evaluate_day(day_rows, load)
    if not math.isclose(before["total_energy_kwh"], after["total_energy_kwh"], rel_tol=1e-10):
        raise AssertionError("Total hospital energy must remain equal in this demo")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()})


def main() -> None:
    config = load_config()
    rows, tasks_by_day = generate_data(config)
    split = config["training_days"] * SLOTS_PER_DAY
    coefficients = fit_linear_regression(rows[:split])
    test_rows = rows[split:]
    predicted = [predict(row, coefficients) for row in test_rows]
    predicted_total = [p + row["baseline_task_kw"] for p, row in zip(predicted, test_rows)]
    yesterday = [rows[split + i - SLOTS_PER_DAY]["actual_total_kw"] for i in range(len(test_rows))]
    actual = [row["actual_total_kw"] for row in test_rows]
    metric = errors(actual, predicted_total)
    yesterday_metric = errors(actual, yesterday)
    predictions = []
    for row, background, total, naive in zip(test_rows, predicted, predicted_total, yesterday):
        predictions.append({
            "timestamp": row["timestamp"],
            "actual_total_kw": row["actual_total_kw"],
            "predicted_background_kw": background,
            "predicted_total_kw": total,
            "yesterday_same_time_kw": naive,
        })

    days, schedules = [], []
    total_before = {"cost_yuan": 0.0, "carbon_kg": 0.0, "total_energy_kwh": 0.0, "grid_energy_kwh": 0.0}
    total_after = dict(total_before)
    total_tasks = 0
    for offset in range(config["days"] - config["training_days"]):
        start = offset * SLOTS_PER_DAY
        day_rows = test_rows[start:start + SLOTS_PER_DAY]
        background_prediction = predicted[start:start + SLOTS_PER_DAY]
        tasks = tasks_by_day[config["training_days"] + offset]
        scheduled_load, chosen_starts = optimize_day(day_rows, tasks, background_prediction, config)
        verify_plan(day_rows, tasks, scheduled_load, chosen_starts, config)
        before = evaluate_day(day_rows, baseline_task_load(tasks))
        after = evaluate_day(day_rows, scheduled_load)
        for key in total_before:
            total_before[key] += before[key]
            total_after[key] += after[key]
        total_tasks += len(tasks)
        days.append({
            "date": day_rows[0]["timestamp"][:10],
            "actual_background": [r["actual_background_kw"] for r in day_rows],
            "critical": [r["critical_kw"] for r in day_rows],
            "hvac": [r["hvac_kw"] for r in day_rows],
            "imaging": [r["imaging_fixed_kw"] for r in day_rows],
            "lighting": [r["lighting_kw"] for r in day_rows],
            "predicted_background": background_prediction,
            "actual_total": before["total_kw"],
            "predicted_total": predicted_total[start:start + SLOTS_PER_DAY],
            "pv_actual": [r["pv_actual_kw"] for r in day_rows],
            "pv_forecast": [r["pv_forecast_kw"] for r in day_rows],
            "tariff": [r["tariff_yuan_per_kwh"] for r in day_rows],
            "carbon": [r["carbon_kg_per_kwh"] for r in day_rows],
            "tasks": tasks,
        })
        for row, bgrid, ogrid, task_power in zip(day_rows, before["grid_kw"], after["grid_kw"], scheduled_load):
            schedules.append({
                "timestamp": row["timestamp"],
                "baseline_total_kw": row["actual_total_kw"],
                "optimized_total_kw": row["actual_background_kw"] + task_power,
                "baseline_grid_kw": bgrid,
                "optimized_grid_kw": ogrid,
                "scheduled_task_kw": task_power,
                "tariff_yuan_per_kwh": row["tariff_yuan_per_kwh"],
                "carbon_kg_per_kwh": row["carbon_kg_per_kwh"],
            })

    report = {
        "data_type": "synthetic",
        "forecast_type": "day-ahead, using simulated weather forecasts and planned occupancy",
        "training_days": config["training_days"],
        "test_days": config["days"] - config["training_days"],
        "model": "linear regression with small ridge penalty",
        "forecast_metrics": metric,
        "yesterday_baseline_metrics": yesterday_metric,
        "baseline": total_before,
        "optimized": total_after,
        "tasks_completed": total_tasks,
        "task_constraint_violations": 0,
        "energy_conservation_passed": True,
        "model_coefficients": coefficients,
        "config": config,
    }
    write_csv(ROOT / "data" / "hospital_energy.csv", rows)
    write_csv(ROOT / "results" / "predictions.csv", predictions)
    write_csv(ROOT / "results" / "schedules.csv", schedules)
    (ROOT / "results" / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {"config": config, "metrics": metric, "naive_metrics": yesterday_metric, "days": days}
    template = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    dashboard = template.replace("__MEDENERGY_DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    (ROOT / "dashboard.html").write_text(dashboard, encoding="utf-8")
    cost_reduction = 100 * (1 - total_after["cost_yuan"] / total_before["cost_yuan"])
    carbon_reduction = 100 * (1 - total_after["carbon_kg"] / total_before["carbon_kg"])
    print("MedEnergy demo generated successfully.")
    print(f"Data: {len(rows)} rows, 15-minute intervals")
    print(f"Chronological split: {config['training_days']} train days / {report['test_days']} test days")
    print(f"Forecast MAE: {metric['mae_kw']:.2f} kW; RMSE: {metric['rmse_kw']:.2f} kW")
    print(f"Yesterday baseline MAE: {yesterday_metric['mae_kw']:.2f} kW")
    print(f"Cost reduction: {cost_reduction:.2f}%; carbon reduction: {carbon_reduction:.2f}%")
    print(f"Tasks completed: {total_tasks}; deadline violations: 0; equal task energy: PASS")
    print(f"Open: {ROOT / 'dashboard.html'}")


if __name__ == "__main__":
    main()
