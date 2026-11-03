"""Rank-based Granger regression tests. Run from the repository root."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2, f

from enhanced_granger_analysis.results.granger_results import GrangerAnalysisResults
from enhanced_granger_analysis.results.statistics import nested_design_geometry


def make_case(use_bias=True, duplicate=True):
	rng=np.random.default_rng(72)
	n=180
	a=rng.normal(size=n)+.5
	b=rng.normal(size=n)-.7
	c=a+b if duplicate else rng.normal(size=n)
	x=np.column_stack([a,b,c])
	y=np.column_stack([.3*a-.2*b+.8*c, .2*a+.6*c])+rng.normal(size=(n,2))*.5
	if use_bias:y+=np.array([1.,-.5])
	result=GrangerAnalysisResults(effects=["y1","y2"],causes=["redundant","pair"],
		mask=np.ones((2,3),bool),cause_slices={"redundant":slice(2,3),"pair":slice(0,2)},
		X_true=x,y_true=y)
	um=np.array([[1,1,1],[1,0,1]],bool)
	for cause in result.causes:
		rm=um.copy();rm[:,result.cause_slices[cause]]=False
		for role,mask in [("unrestricted",um),("restricted",rm)]:
			pred=np.empty_like(y);weights=np.zeros((2,3))
			for j in range(2):
				d=x[:,mask[j]]
				if use_bias:d=np.column_stack([np.ones(n),d])
				coef=np.linalg.lstsq(d,y[:,j],rcond=None)[0]
				pred[:,j]=d@coef
				weights[j,mask[j]]=coef[int(use_bias):]
			setter=result.set_unrestricted_snapshot if role=="unrestricted" else result.set_restricted_snapshot
			setter(cause,pred,weights,active_slope_mask=mask,bias_enabled=use_bias,
				   metadata={"coefficient_space":"raw","fit_performed":True})
	result.set_dof_method("rank")
	return result


@pytest.mark.parametrize("use_bias",[False,True])
def test_rank_counts_redundant_columns(use_bias):
	result=make_case(use_bias)
	np.testing.assert_array_equal(result.n_unrestricted_parameters("redundant"),[2+int(use_bias)]*2)
	np.testing.assert_array_equal(result.n_restrictions_for("redundant"),[0,1])
	np.testing.assert_array_equal(result.n_restrictions_for("pair"),[1,1])


@pytest.mark.parametrize("use_bias",[False,True])
@pytest.mark.parametrize("duplicate",[False,True])
def test_rank_metrics_match_reference_spaces(use_bias,duplicate):
	result=make_case(use_bias,duplicate)
	W=result.evaluate("wald_test",restore_removed=False).data.to_numpy()
	F=result.evaluate("f_statistic",restore_removed=False).data.to_numpy()
	LR=result.evaluate("likelihood_ratio",restore_removed=False).data.to_numpy()
	PW=result.evaluate("wald_test_p_value",restore_removed=False).data.to_numpy()
	PF=result.evaluate("f_statistic_p_value",restore_removed=False).data.to_numpy()
	for col,cause in enumerate(result.causes):
		q=result.n_restrictions_for(cause)
		k=result.n_unrestricted_parameters(cause)
		nu=result.n_samples-k
		for j in range(2):
			if q[j]==0:
				assert np.isnan(W[j,col]) and np.isnan(F[j,col])
				assert np.isnan(PW[j,col]) and np.isnan(PF[j,col])
			else:
				assert W[j,col]==pytest.approx(q[j]*F[j,col],rel=1e-8,abs=1e-9)
				assert LR[j,col]==pytest.approx(result.n_samples*np.log1p(q[j]*F[j,col]/nu[j]),rel=1e-8,abs=1e-9)
				assert PW[j,col]==pytest.approx(chi2.sf(W[j,col],q[j]))
				assert PF[j,col]==pytest.approx(f.sf(F[j,col],q[j],nu[j]))


def test_rank_wald_does_not_use_restricted_predictions():
	result=make_case()
	before=result.rank_wald_per_output("pair")
	result.get_snapshot("pair","restricted").predictions[:]=12345.
	result.get_snapshot("pair","restricted").metadata["fit_performed"]=False
	after=result.rank_wald_per_output("pair")
	np.testing.assert_allclose(before,after)


def test_rank_wald_is_invariant_to_nullspace_coefficients():
	result=make_case()
	before=result.rank_wald_per_output("pair")
	# First output: x3=x1+x2, so this changes coefficients but not predictions.
	u=result.get_snapshot("pair","unrestricted")
	u.weights[0]+=np.array([10.,10.,-10.])
	np.testing.assert_allclose(before,result.rank_wald_per_output("pair"))


def test_rank_option_invalidates_cache_and_can_be_disabled():
	result=make_case()
	result.rank_diagnostics()
	result._metric_cache["test"]=object()
	assert result._rank_cache
	result.set_dof_method("structural")
	assert not result._rank_cache and not result._metric_cache
	np.testing.assert_array_equal(result.n_unrestricted_parameters("pair"),[4,3])


def test_snapshot_replacement_invalidates_rank_cache():
	result=make_case()
	result.rank_diagnostics()
	snapshot=result.get_snapshot("pair","restricted")
	result.register_snapshot("pair","restricted",snapshot)
	assert not result._rank_cache


def test_near_collinearity_has_configurable_cutoff():
	rng=np.random.default_rng(19)
	a=rng.normal(size=100)
	u=np.column_stack([a,a+1e-10*rng.normal(size=100)])
	r=u[:,:1]
	assert nested_design_geometry(u,r,rcond=1e-14)["q"]==1
	assert nested_design_geometry(u,r,rcond=1e-8)["q"]==0


@pytest.mark.parametrize("rcond",[-1.,1.,np.nan,True])
def test_invalid_tolerance(rcond):
	result=make_case()
	with pytest.raises(ValueError):result.set_dof_method("rank",rank_rcond=rcond)


def test_intercept_collinear_with_constant_feature_is_not_double_counted():
	ones=np.ones((80,1))
	g=nested_design_geometry(np.column_stack([ones,ones]),ones)
	assert g["rank_u"]==1 and g["rank_r"]==1 and g["q"]==0


@pytest.mark.parametrize("use_bias",[False,True])
def test_rank_f_against_statsmodels_for_redundant_design(use_bias):
	sm=pytest.importorskip("statsmodels.api")
	result=make_case(use_bias)
	actual=result.evaluate("f_statistic",restore_removed=False).data.to_numpy()
	for col,cause in enumerate(result.causes):
		u=result.get_snapshot(cause,"unrestricted")
		r=result.get_snapshot(cause,"restricted")
		q=result.n_restrictions_for(cause)
		for j in range(2):
			if q[j]==0:continue
			du=result.X_true[:,u.active_slope_mask[j]]
			dr=result.X_true[:,r.active_slope_mask[j]]
			if use_bias:
				du=np.column_stack([np.ones(result.n_samples),du])
				dr=np.column_stack([np.ones(result.n_samples),dr])
			fu=sm.OLS(result.y_true[:,j],du).fit()
			fr=sm.OLS(result.y_true[:,j],dr).fit()
			expected=fu.compare_f_test(fr)
			assert actual[j,col]==pytest.approx(expected[0],rel=1e-8,abs=1e-9)
