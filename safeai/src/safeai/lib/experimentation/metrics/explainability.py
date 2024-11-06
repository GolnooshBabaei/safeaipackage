from pandas import DataFrame
from pydantic import computed_field
from safeai.lib.experimentation.base import SafeAIMetric


class Explainability(SafeAIMetric):
    """_summary_: A metric to evaluate the explainability of a model"""

    def compute_single_variable_rge(self, variable: str) -> float:
        """_summary_: Computes the RGE for a single variable"""
        _xtest = self.experiment_job.xtest.copy()
        if variable in _xtest.columns:
            if self.experiment_job.encodes:
                if (
                    any([variable.startswith(i) for i in self.experiment_job.encodes])
                    or self.experiment_job.xtrain[variable].nunique() == 2
                ):
                    _xtest[variable] = self.experiment_job.xtrain[variable].mode()[0]
            else:
                _xtest[variable] = self.experiment_job.xtrain[variable].mean()
            return self.get_rga(
                self.experiment_job.predict_proba(_xtest),
                self.experiment_job.predictions_test["y"],
            )
        raise ValueError(f"Variable '{variable}' not found in the dataset")

    @computed_field
    @property
    def compute_rge(self) -> dict:
        """_summary_: Computes the RGE for a group of variables"""
        return {
            x: self.compute_single_variable_rge(x)
            for x in self.experiment_job.xtrain.columns
        }

    @computed_field
    @property
    def compute_group_rge(self) -> DataFrame:
        """_summary_: Computes the PDP for a single variable"""
        raise NotImplementedError("This method is not implemented")
