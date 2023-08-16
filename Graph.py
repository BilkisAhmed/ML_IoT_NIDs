import numpy as np
import matplotlib.pyplot as plt

X = ['Log_Reg', 'KNN', 'Random_Forest', 'SVMClassifier']
feature_39 = [0.9819, 0.999, 0.997, 0.972]
feature_9 = [0.9627,0.998, 0.998, 0.9843]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("Recall")
plt.title("ML Detection Rate")
plt.legend()
plt.tight_layout()
plt.savefig('Sup_Auc.png')
plt.show()


X = ['Log_Reg', 'KNN', 'Random_Forest', 'SVMClassifier']
feature_39 = [0.1723, 0.0012, 0.031, 0.1687]
feature_9 = [0.1705,0.0013, 0.0095, 0.1728]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("False Positive Rate")
plt.title("ML Models False Alarm Rate")
plt.legend()
plt.tight_layout()
plt.savefig('Sup_fpr.png')
plt.show()


X = ['Log_Reg', 'KNN', 'Random_Forest', 'SVMClassifier']
feature_39 = [0.008619,215.97, 3.067, 333.91]
feature_9 = [0.0029,2.16, 2.933, 152.87]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("Detection Time")
plt.title("ML Models Detection Time(s)")
plt.legend()
plt.tight_layout()
plt.savefig('Sup_time.png')
plt.show()



X = ['OC_SVM', 'Iforest', 'Elliptic', 'Local_outlierF']
feature_39 = [1.0, 0.698, 0.8798, 0.894]
feature_9 = [0.998,0.9988, 0.998, 0.871]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("Recall")
plt.title("ML Models Detection Rate")
plt.legend()
plt.tight_layout()
plt.savefig('unSup_auc.png')
plt.show()


X =['OC_SVM', 'Iforest', 'Elliptic', 'Local_outlierF']
feature_39 = [0.6354, 0.5023, 0.501, 0.051]
feature_9 = [0.5365,0.4975, 0.5009, 0.04439]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("False Positive Rate")
plt.title("ML Models False Alarm Rate")
plt.legend()
plt.tight_layout()
plt.savefig('unSup_fpr.png')
plt.show()

X = ['OC_SVM', 'Iforest', 'Elliptic', 'Local_outlierF']
feature_39 = [0.2193,0.8835, 0.203, 13.31]
feature_9 = [0.0755,0.9036,0.0233, 3.42]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2,feature_39, 0.4,label='feature_39')
plt.bar(X_axis + 0.2, feature_9, 0.4,label='feature_9')

plt.xticks(X_axis, X)
plt.xlabel("Models")
plt.ylabel("Detection Time(s)")
plt.title("ML Models Detection Time")
plt.legend()
plt.tight_layout()
plt.savefig('unSup_time.png')
plt.show()
