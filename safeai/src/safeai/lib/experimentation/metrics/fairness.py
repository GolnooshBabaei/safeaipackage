from pandas import DataFrame, Series
from pydantic import computed_field
from safeai.enums import PredictionType
from safeai.lib.experimentation.base import SafeAIMetric


class Fairness(SafeAIMetric):
    """_summary_: A metric to evaluate the fairness of a model"""

    def _single_group_rga_parity(
        self,
        x:DataFrame,
        y:Series,
        protected_variable:str,
        group_name:str
    ) -> float:
        """_summary_: Computes RGA-based imparity MEASURE. """
        group_mask = x[protected_variable] == group_name

        if not group_mask.any():
            raise ValueError(f"Group '{group_name}' not found in '{protected_variable}'")

        try:
            if self.experiment_job.prediction_type == PredictionType.REGRESSION:
                predictions = self.experiment_job.predict_proba(x[group_mask])
            else:
                predictions = self.experiment_job.fit.predict(x[group_mask])
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
        return self.get_rga(y[group_mask], predictions)


    def _single_variable_rga_parity(
        self,
        x:DataFrame,
        y:Series,
        protected_variable:str
    ) -> float:
        """_summary_: Computes RGA-based imparity MEASURE. """
        _rgas = list(
                map(
                    lambda g: self._single_group_rga_parity(x, y, protected_variable, g),
                    x[protected_variable].unique()
                )
            )
        return sum(_rgas) / len(_rgas)

    @computed_field
    @property
    def compute_rga_parity(self) -> float | None:
        """_summary_: Computes the RGA Parity"""
        if self.experiment_job.protected_variables:
            rgas = list(
                map(
                    lambda x: self._single_variable_rga_parity(
                        self.experiment_job.xtest,
                        self.experiment_job.ytest.y,
                        x
                    ),
                    self.experiment_job.protected_variables
                )
            )
            if len(rgas) == 1:
                return rgas[0]
            return max(rgas) - min(rgas)
        return None
