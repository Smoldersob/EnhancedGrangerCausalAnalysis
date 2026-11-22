from .layout import RegularizationLayout
from .prepared import PreparedRegularizerData


def validate_regularization_layout(layout: RegularizationLayout) -> None:
    if not isinstance(layout, RegularizationLayout):
        raise TypeError("layout must be a RegularizationLayout instance.")


def validate_prepared_regularizer_data(data: PreparedRegularizerData) -> None:
    if not isinstance(data, PreparedRegularizerData):
        raise TypeError("data must be a PreparedRegularizerData instance.")
