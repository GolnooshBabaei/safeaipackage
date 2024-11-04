from numpy import ceil
from pandas import DataFrame
from pydantic import computed_field
from safeai.lib.experimentation.base import SafeAIMetric


class Robustness(SafeAIMetric):
    """_summary_: A metric to evaluate the robustness of a model"""

    def _pertub_single_variable(
        self,
        data:DataFrame,
        protected_variable:str
    ) -> DataFrame:
        """_summary_: Pertub the data"""
        _data = data.copy().sort_values(by=protected_variable).reset_index(drop=True)
        
        lower_tail = _data.iloc[:int(ceil(self.experiment_job.perturbation * len(data)))]
        upper_tail = _data.iloc[int(ceil(1-self.experiment_job.perturbation * len(data))):]
        for j in range(min(len(lower_tail), len(upper_tail))):
            _data.at[j, protected_variable] = upper_tail.iloc[j][protected_variable]
            _data.at[len(_data) - j - 1, protected_variable] = lower_tail.iloc[j][protected_variable]
        
        return _data


    @computed_field
    @property
    def compute_rgr(self) -> dict:
        """_summary_: Computes the Full Single RGR"""
        return {
            x: self.get_rga(
                    self.experiment_job.predict_proba(
                        self._pertub_single_variable(
                            self.experiment_job.xtest, x
                        )
                    ), self.experiment_job.predictions_test["y"]
                )
            for x in self.experiment_job.xtrain.columns
        }
