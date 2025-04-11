# SAFE AI

The increasing widespread of Artificial Intelligence (AI) applications implies the formalisa-
tion of an AI risk management model which needs methodological guidelines for an effec-
tive implementation. The base of this package is the S.A.F.E. AI metrics in the “Rank Graduation
Box” proposed in [Babaei et al. 2024](https://www.sciencedirect.com/science/article/pii/S0957417424021067). The use of the term “box” is motivated by the need of emphasizing that our proposal is
always in progress so that, like a box, it can be constantly filled by innovative tools addressed
to the measurement of the new future requirements necessary for the safety condition of
AI-systems.

This package includes different modules, each proposed to measure one of the considered AI ethics in this project (Accuacy, Explainability, Fairness and Robustness). The core metric proposed in this framework is Rank Graduation Accuracy (RGA) which is an extention of AUC. However, RGA is not only applicable for classification models but also for regression models. The modules available in this package are as follows:

### Core

This module is the core of safeaipackage. In particular, using rga function from this module, it is possible to calculate __Rank Graduation Accuracy (RGA)__ metric which is the base concept for the calculation of the other metrics.

__Functions:__

rga(y, yhat)

This function compares ranks of the actual values with ranks of the predicted values. Therefore, in the case of the equality between the ranks, RGA is equal to one (the best case). In general, RGA gets values between 0 and 1.

- **Parameters**:
    - `y`: Actual response values 
    - `yhat`: Predicted probabilities

- **Returns**: RGA value


### check_explainability

This module includes the compute_rge_values function to measure contribution of the variables (individually or in a group) to the model. In particular, using the function available in this module, it is possible to calculate __Rank Graduation Explainability (RGE)__ metric which is base on RGA.  

__Functions:__

compute_rge_values(xtrain, 
                xtest,
                yhat,
                model,  
                variables, 
                group= False)

This function calculates RGE for each single considered variable, individually or in a group based on the provided value for group parameter. In particular, this function compares ranks of the predicted values including all the variables with the ranks of the predicted values by the model excluding the effect of the selected variable(s). When RGE is equal to 1, it shows a variable with a high contribution to the model. while when it is equal to 0, there is no contribution to the predictions.

- **Parameters**:
    - `xtrain`: Train data
    - `xtest`: Test data
    - `yhat`: Predicted probabilities  
    - `model`: A trained model, which could be a classifier or regressor.
    - `variables`: List of variables 
    - `group`: A boolean parameter. If True, calculate RGE for the group of variables as a whole; otherwise, calculate for each variable.
    
- **Returns**: RGE 


### check_fairness

This module provides model imparity analysis. Using the function in this module, the difference between RGA of the model in the two protected groups is measured. The protected variable is supposed to be a binary categorical variable (referring to privileged and unprivileged groups).

__Functions:__

compute_rga_parity(xtrain, xtest, ytest, yhat, model, protectedvariable)

This function calculates RGA values for the protected groups considering the given protected variable. 

- **Parameters**:
    - `xtrain`: Train data
    - `xtest`: Test data
    - `ytest`: Actual response values
    - `yhat`: Predicted probabilities  
    - `model`: A trained model, which could be a classifier or regressor.
    - `protectedvariable`: Protected (sensitive) variable which is a binary categorical variable
    
    
- **Returns**: Difference between the RGA values in each protected group


### check_robustness

This module includes a function to measure robustness of the model towards the perturbations applied to the variables. In particular, using the function available in check_robustness module, it is possible to calculate __Rank Graduation Robustness (RGR)__ metric which is base on RGA.  

__Functions:__

compute_rgr_values(xtest, yhat, model, variables, perturbation_percentage= 0.05)

This function calculates RGR for each single considered variable. In other words, this function compares ranks of the predicted values by the model including the original values of the variables with the ranks of the predicted values by the model including the selected perturbed variable. When RGR is equal to 1, it shows that the model is completely robust to the variable perturbations.When RGR is equal to 0, model is not robust.

- **Parameters**:
    - `xtest`: Test data
    - `yhat`: Predicted probabilities  
    - `model`: A trained model, which could be a classifier or regressor.
    - `variables`: List of variables 
    - `perturbation_percentage`: The percentage for perturbation process
    
- **Returns**: RGR 


# Install

Simply use:

pip install safeaipackage



# Example

In the folder "examples", we present a classification and a regression problem applied to the [employee dataset](https://search.r-project.org/CRAN/refmans/stima/html/employee.html).



# Support

If you need help or have any questions, the first step should be to take a look at the docs. If you can't find an answer, please open an issue on GitHub, or send an email to golnoosh.babaei@unipv.it. 



# Citations

The proposed measures in this package came primarily out of research by 
[Paolo Giudici](https://www.linkedin.com/in/paolo-giudici-60028a/), [Emanuela Raffinetti](https://www.linkedin.com/in/emanuela-raffinetti-a3980215/), 
and [Golnoosh Babaei](https://www.linkedin.com/in/golnoosh-babaei-990077187/) in the [Statistical laboratory](https://sites.google.com/unipv.it/statslab-pavia/home?authuser=0) 
at the University of Pavia. 
This package is based on the following papers. If you use safeaipackage in your research we would appreciate a citation to our papers:
* [Babaei, G., Giudici, P., & Raffinetti, E. (2024). A Rank Graduation Box for SAFE AI. Expert Systems with Applications, 125239.](https://doi.org/10.1016/j.eswa.2024.125239)
* [Giudici, P., & Raffinetti, E. (2024). RGA: a unified measure of predictive accuracy. Advances in Data Analysis and Classification, 1-27.](https://link.springer.com/article/10.1007/s11634-023-00574-2)
* [Raffinetti, E. (2023). A rank graduation accuracy measure to mitigate artificial intelligence risks. Quality & Quantity, 57(Suppl 2), 131-150.](https://link.springer.com/article/10.1007/s11135-023-01613-y)
