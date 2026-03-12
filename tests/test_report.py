"""
Tests for FrailtyReport.

We test that the HTML output is structurally valid and contains
key content, without actually rendering in a browser.
"""

from __future__ import annotations

import pytest

from insurance_recurrent import RecurrentEventSimulator, SharedFrailtyModel, FrailtyReport


@pytest.fixture(scope="module")
def report_fixture(tmp_path_factory):
    sim = RecurrentEventSimulator(n_policies=100, theta=0.4, baseline_rate=0.3, seed=88)
    data = sim.simulate()
    m = SharedFrailtyModel(max_iter=15)
    m.fit(data)
    report = FrailtyReport(m, data, model_name="Test Fleet Model")
    tmp_dir = tmp_path_factory.mktemp("reports")
    return report, data, tmp_dir


class TestFrailtyReportRender:
    def test_render_returns_string(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert isinstance(html, str)

    def test_html_has_doctype(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert "<!DOCTYPE html>" in html

    def test_model_name_in_output(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert "Test Fleet Model" in html

    def test_theta_in_output(self, report_fixture):
        report, data, _ = report_fixture
        _, model = report.model, report.model
        html = report.render()
        assert "theta" in html.lower()

    def test_policy_count_in_output(self, report_fixture):
        report, data, _ = report_fixture
        html = report.render()
        assert str(data.n_policies) in html

    def test_version_in_output(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert "0.1.0" in html

    def test_contains_credibility_section(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert "credibility" in html.lower()

    def test_contains_decile_section(self, report_fixture):
        report, _, _ = report_fixture
        html = report.render()
        assert "decile" in html.lower()


class TestFrailtyReportSave:
    def test_save_creates_file(self, report_fixture, tmp_path):
        report, _, _ = report_fixture
        out_path = str(tmp_path / "test_report.html")
        report.save(out_path)
        import os
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000  # non-trivial file

    def test_saved_file_is_valid_html(self, report_fixture, tmp_path):
        report, _, _ = report_fixture
        out_path = str(tmp_path / "test_report2.html")
        report.save(out_path)
        with open(out_path) as f:
            content = f.read()
        assert "<html" in content
        assert "</html>" in content


class TestFrailtyReportUnfitted:
    def test_unfitted_model_raises(self):
        sim = RecurrentEventSimulator(n_policies=50, seed=1)
        data = sim.simulate()
        m = SharedFrailtyModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            FrailtyReport(m, data)


class TestFrailtyReportWithCovariates:
    def test_report_shows_coefficients(self):
        sim = RecurrentEventSimulator(
            n_policies=100, theta=0.4, coef={"risk_band": 0.5}, seed=7
        )
        data = sim.simulate()
        m = SharedFrailtyModel(max_iter=10)
        m.fit(data, covariates=["risk_band"])
        report = FrailtyReport(m, data)
        html = report.render()
        assert "Coefficients" in html
        assert "risk_band" in html
