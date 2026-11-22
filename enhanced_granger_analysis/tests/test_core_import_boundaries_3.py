from enhanced_granger_analysis.preprocessing.lag.artifacts import LagFittedConfig
from enhanced_granger_analysis.api.outputs import SimpleGrangerOutput
from enhanced_granger_analysis.configuration.training_spec import TrainingConfig
from enhanced_granger_analysis.preprocessing.scaling.protocol import Scaler
from enhanced_granger_analysis.core.data.analysis_roles import VariableRoles


def test_new_import_boundaries_resolve_expected_objects():
    assert LagFittedConfig.__name__ == "LagFittedConfig"
    assert SimpleGrangerOutput.__name__ == "SimpleGrangerOutput"
    assert TrainingConfig.__name__ == "TrainingConfig"
    assert Scaler.__name__ == "Scaler"
    assert VariableRoles.__name__ == "VariableRoles"
