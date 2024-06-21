import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classify import ViolenceClass
import numpy as np

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)

evaluator = ViolenceClass()

# Sample data

real_labels, predicted_labels ,predicted_probs = evaluator.make_predictions()
predicted_probs = [softmax(output)[1] for output in predicted_probs]
# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(real_labels, predicted_probs)

# Calculate the AUC (Area Under the Curve)
roc_auc = roc_auc_score(real_labels, predicted_probs)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for testset gauss')
plt.legend(loc='lower right')
plt.show()
