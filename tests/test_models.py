import json
from pathlib import Path

import numpy as np

from mmc_model.data import find_attachment2, load_attachment2, split_topologies
from mmc_model.optimization import (
    Design,
    clip_halfplane,
    evaluate_grid,
    nondominated_mask,
    polygon_area,
    weight_acceptability,
    worst_case_box,
)
from mmc_model.q1 import coolant_energy_balance, dual_manifold_flow_balance, no_pin_pressure_fit
from mmc_model.rsm import CubicRSM, total_degree_powers
from mmc_model.validation import local_beta_eta_block_holdout

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = find_attachment2(ROOT)


def test_attachment_structure():
    frame = load_attachment2(WORKBOOK)
    no_pin, pin = split_topologies(frame)
    assert len(no_pin) == 4
    assert len(pin) == 80
    assert pin.groupby(["beta", "eta", "N"]).size().eq(1).all()


def test_cubic_interpolates_thermal_resistance_nearly_exactly():
    _, pin = split_topologies(load_attachment2(WORKBOOK))
    model = CubicRSM.fit(pin.beta, pin.eta, pin.N, pin[["R", "P", "U"]])
    pred = model.predict(pin.beta.to_numpy(), pin.eta.to_numpy(), pin.N.to_numpy())
    rmse = np.sqrt(np.mean((pred - pin[["R", "P", "U"]].to_numpy()) ** 2, axis=0))
    assert len(total_degree_powers(3)) == 20
    assert rmse[0] < 1e-6
    assert rmse[1] < 1e-3
    assert rmse[2] < 5e-4


def test_q1_energy_and_pressure_scale():
    no_pin, _ = split_topologies(load_attachment2(WORKBOOK))
    balance = coolant_energy_balance()
    fit = no_pin_pressure_fit(no_pin)
    assert np.isclose(balance["heat_W"], 36.0)
    assert np.isclose(balance["mean_coolant_rise_K"], 8.6083, atol=1e-3)
    assert fit["r2"] > 0.999


def test_dual_manifold_flow_balance():
    result = dual_manifold_flow_balance(np.array([0.20, 0.35, 0.15, 0.30]))
    assert np.allclose(result["inlet"], [1.00, 0.80, 0.45, 0.30, 0.00])
    assert np.allclose(result["outlet"], [0.00, 0.20, 0.55, 0.70, 1.00])
    assert np.allclose(result["inlet"] + result["outlet"], 1.00)

    with np.testing.assert_raises(ValueError):
        dual_manifold_flow_balance(np.array([0.4, -0.1, 0.7]))


def test_halfplane_clipping():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    clipped = clip_halfplane(triangle, 1.0, 1.0, 0.5)
    assert np.isclose(polygon_area(clipped), 0.125)


def test_nondominated_filter_rejects_equal_third_objective():
    values = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    mask = nondominated_mask(values)
    assert np.array_equal(mask, [True, False, False])


def test_weight_regions_use_full_simplex_area():
    values = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )
    areas, _ = weight_acceptability(values)
    assert np.allclose(areas[:3], np.full(3, 1 / 3), atol=1e-10)
    assert np.isclose(areas[3], 0.0, atol=1e-10)
    assert np.isclose(areas.sum(), 1.0, atol=1e-10)


def test_integer_n_grid_contains_temperature_uniformity_ideal():
    _, pin = split_topologies(load_attachment2(WORKBOOK))
    model = CubicRSM.fit(pin.beta, pin.eta, pin.N, pin[["R", "P", "U"]])
    _, values = evaluate_grid(model, beta_points=101, eta_points=76)
    assert values[:, 2].min() < 0.77084


def test_q4_stability_checks_retain_n4_neighborhood():
    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    even = results["q4"]["even_N_check"]
    fine = results["q4"]["fine_grid_check"]
    for check in (even, fine):
        assert check["robust_design"]["N"] == 4
        assert np.isclose(check["robust_design"]["eta"], 4.5)
        assert 0.22 <= check["robust_design"]["beta"] <= 0.225
        assert check["acceptability_by_N"]["4"] > 0.45


def test_q3_fine_grid_check_retains_n6_neighborhood():
    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    fine = results["q3"]["fine_grid_check"]
    design = fine["chebyshev"]["design"]
    assert design["N"] == 6
    assert np.isclose(design["eta"], 4.5)
    assert 0.22 <= design["beta"] <= 0.23


def test_local_beta_eta_block_holdout_remains_accurate():
    _, pin = split_topologies(load_attachment2(WORKBOOK))
    rmse = local_beta_eta_block_holdout(pin)
    assert rmse[0] < 1e-6
    assert rmse[1] < 5e-4
    assert rmse[2] < 5e-4


def test_q4_cluster_is_defined_by_main_grid_then_recomputed():
    results = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    cluster = results["q4"]["stable_cluster"]
    assert np.allclose(cluster["definition"]["beta"], [0.215, 0.224])
    assert cluster["definition"]["eta"] == 4.5
    assert cluster["definition"]["N"] == [3, 4]
    assert cluster["main_grid"] > 0.51
    assert cluster["even_N_only"] > 0.48
    assert cluster["fine_grid"] > 0.49


def test_overall_maximum_is_componentwise_maximum():
    _, pin = split_topologies(load_attachment2(WORKBOOK))
    model = CubicRSM.fit(pin.beta, pin.eta, pin.N, pin[["R", "P", "U"]])
    design = Design(0.22494283335071952, 4.5, 6)
    baseline = model.predict(design.beta, design.eta, design.n_rows)
    result = worst_case_box(model, design, 0.05, baseline)
    controlling_response = int(np.argmax(result["relative_degradation"]))
    overall_maximum = float(np.max(result["relative_degradation"]))
    assert controlling_response == 1
    assert 0.07 < overall_maximum < 0.072
