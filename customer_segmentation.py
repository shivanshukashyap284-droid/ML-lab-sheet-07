import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

df=pd.read_csv('customer_data.csv').drop_duplicates()
features=['Age','AnnualIncome','SpendingScore','PurchaseFrequency']
df[features]=df[features].apply(pd.to_numeric,errors='coerce'); df=df.dropna(subset=features)
X=StandardScaler().fit_transform(df[features])
K=range(2,9); inertias=[]; scores=[]
for k in K:
 m=KMeans(n_clusters=k,random_state=42,n_init=10); y=m.fit_predict(X); inertias.append(m.inertia_); scores.append(silhouette_score(X,y))
best_k=list(K)[int(np.argmax(scores))]; print('Optimal K:',best_k)
plt.figure(figsize=(7,4)); plt.plot(list(K),inertias,marker='o'); plt.xlabel('K'); plt.ylabel('Inertia'); plt.title('Elbow Method'); plt.tight_layout(); plt.savefig('01_elbow_plot.png',dpi=150); plt.show()
plt.figure(figsize=(7,4)); plt.plot(list(K),scores,marker='o'); plt.xlabel('K'); plt.ylabel('Silhouette Score'); plt.title('Silhouette Analysis'); plt.tight_layout(); plt.savefig('02_silhouette_plot.png',dpi=150); plt.show()
m=KMeans(n_clusters=best_k,random_state=42,n_init=10); df['KMeansCluster']=m.fit_predict(X)
plt.figure(figsize=(8,6)); sns.scatterplot(data=df,x='AnnualIncome',y='SpendingScore',hue='KMeansCluster',s=70); plt.title('Customer Segmentation using K-Means'); plt.tight_layout(); plt.savefig('03_kmeans_clusters.png',dpi=150); plt.show()
profile=df.groupby('KMeansCluster')[features].mean().round(2); print(profile); profile.to_csv('cluster_profile.csv')
Z=linkage(X,method='ward'); plt.figure(figsize=(10,5)); dendrogram(Z,truncate_mode='lastp',p=25); plt.title('Hierarchical Clustering Dendrogram'); plt.tight_layout(); plt.savefig('04_dendrogram.png',dpi=150); plt.show()
h=fcluster(Z,t=best_k,criterion='maxclust')-1; df['HierarchicalCluster']=h
print('K-Means Silhouette:',round(silhouette_score(X,df.KMeansCluster),4)); print('Hierarchical Silhouette:',round(silhouette_score(X,h),4))
df.to_csv('customer_segmentation_results.csv',index=False)
