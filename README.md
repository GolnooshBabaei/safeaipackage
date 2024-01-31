# SAFE AI

To fulfill trustworthiness, Artificial Intelligence (AI) methods need to be safe. SAFE referes to an AI based model for which risk measures are controlled. 
The general risks of AI can be considered in the following categories: Privacy, Robustness, Accuracy, Fairness, and Explainability.
In other words, a safe application of AI must satisfy these basic key-principles. First of all, it should be accurate and predicts the target variable close
to the actual values. However, in addition to accuracy, it needs to be explainable because AI approaches are complex and understanding the reasons behind their
results is difficult so to improve trustworthiness we need explainable methods. These models should be fair as well because automated decision-making models are
sometimes biased against some special observations. Hence, we need to be careful about fairness in models. Robustness is another significant aspect that should
be considered when measuring risks of AI because sometimes some small changes in a variable could lead to very different results that leads to risks of the model.
In some cases, it is needed to remove information of some instances so we need to evaluate how model predictions changes when information are removed that refers
to the privacy risk category. 

# Install
safeaipackage can be installed from TestPyPI as follows:

pip install -i https://test.pypi.org/simple/safeaipackage==0.0.1


# Citations
The algorithms and visualizations used in this package came primarily out of research by 
[Paolo Giudici](https://www.linkedin.com/in/paolo-giudici-60028a/), [Emanuela Raffinetti](https://www.linkedin.com/in/emanuela-raffinetti-a3980215/), 
and [Golnoosh Babaei](https://www.linkedin.com/in/golnoosh-babaei-990077187/) in the [Statistical laboratory](https://sites.google.com/unipv.it/statslab-pavia/home?authuser=0) 
at the University of Pavia. If you use safe_ai package in your research we would appreciate a citation to the appropriate paper(s):
* For the RGA measure introduced in "check_accuracy" module, you can read/cite [this paper](https://link.springer.com/article/10.1007/s11135-023-01613-y)

