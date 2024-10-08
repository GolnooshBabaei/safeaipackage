# SAFE AI

The increasing widespread of Artificial Intelligence (AI) applications implies the formalisa-
tion of an AI risk management model which needs methodological guidelines for an effec-
tive implementation. To fill the gap, [Giudici and Raffinetti (2023)](https://www.sciencedirect.com/science/article/pii/S1544612323004609) 
introduced a S.A.F.E.
risk management model which derives from the proposed international regulations four main
compliance principles: Security, Accuracy, Fairness and Explainability, that can be measured
for any AI application. The primary motivation for developing safeaipackage is providing a
unified framework that can evaluate these AI risks.
The important aspect of the S.A.F.E. model is that it proposes metrics that are interrelated,
are standardised, and have a common mathematical root: the Lorenz Zonoid tool (see, e.g. [Koshevoy and Mosler 1996](https://www.tandfonline.com/doi/abs/10.1080/01621459.1996.10476955); [Lorenz 1905](https://link.springer.com/article/10.1007/s11135-023-01613-y)). 
Despite the advantages deriving from the S.A.F.E
approach, it suffers from being computationally intensive as it requires the construction of
all the models’ configurations. Moreover, the recent regulatory debate has further expanded
the notion of Security, distinguishing the internal resilience of an AI system: its robustness;
from the external resilience of the ecosystem which surrounds it: environmental, social and
governance sustainability see, e.g. [International Standard Organisation 2023](https://www.iso.org/standard/77304.html).
In line with this evolution and the need of providing a unified and computationally efficient
method, in this contribution we suggest to combine different statistical metrics able to measure
the Security (robustness), Accuracy, Fairness and Explainability of highly complex machine
learning models. We remark that the definition of robustness that we adopt in this paper
derives from that being employed by AI regulators and standard setters, for which an AI
system should achieve an appropriate level of robustness, to be resilient to internal anomalies
and external attacks. For consistency, the proposed measures will be based on the Lorenz and
concordance curves (see, e.g. [Giudici and Raffinetti 2011](https://www.sciencedirect.com/science/article/pii/S0167715210002816?casa_token=vmope_BDFxcAAAAA:82Klf9ITRpkb7580mnvxfebtLu-SaTcuhpJKnqq6OeF3NtW-xmy5acHsUUJuhGUzkALZUBYX1g)). This will allow to
integrate all measures into an agnostic score that can be employed to assess the trustworthiness
of any AI application.

This S.A.F.E. approach is based on “Rank Graduation
Box” proposed in [Babaei et al. 2024](https://www.sciencedirect.com/science/article/pii/S0957417424021067). The use of the term “box” is motivated by the need of emphasizing that our proposal is
always in progress so that, like a box, it can be constantly filled by innovative tools addressed
to the measurement of the new future requirements necessary for the safety condition of
AI-systems.


# Install

Simply use:

pip install safeaipackage


# Example

On GitHub, in the folder "examples", we present a classification and a regression problem applied to the [employee dataset](https://search.r-project.org/CRAN/refmans/stima/html/employee.html).



# Citations

The proposed measures in this package came primarily out of research by 
[Paolo Giudici](https://www.linkedin.com/in/paolo-giudici-60028a/), [Emanuela Raffinetti](https://www.linkedin.com/in/emanuela-raffinetti-a3980215/), 
and [Golnoosh Babaei](https://www.linkedin.com/in/golnoosh-babaei-990077187/) in the [Statistical laboratory](https://sites.google.com/unipv.it/statslab-pavia/home?authuser=0) 
at the University of Pavia. 
This package is based on the following papers. If you use safeaipackage in your research we would appreciate a citation to our papers:
* [Babaei, G., Giudici, P., & Raffinetti, E. (2024). A Rank Graduation Box for SAFE AI. Expert Systems with Applications, 125239.](https://doi.org/10.1016/j.eswa.2024.125239)
* [Giudici, P., & Raffinetti, E. (2024). RGA: a unified measure of predictive accuracy. Advances in Data Analysis and Classification, 1-27.](https://link.springer.com/article/10.1007/s11634-023-00574-2)
* [Raffinetti, E. (2023). A rank graduation accuracy measure to mitigate artificial intelligence risks. Quality & Quantity, 57(Suppl 2), 131-150.](https://link.springer.com/article/10.1007/s11135-023-01613-y)
