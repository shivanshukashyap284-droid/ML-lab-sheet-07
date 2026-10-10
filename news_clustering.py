import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

df=pd.read_csv('news_articles.csv').drop_duplicates(subset=['Text']).copy(); df['Text']=df.Text.fillna('').str.lower().str.replace(r'\s+',' ',regex=True).str.strip(); df=df[df.Text.str.len()>0].reset_index(drop=True)
vec=TfidfVectorizer(stop_words='english',max_features=1000,ngram_range=(1,2)); X=vec.fit_transform(df.Text); print('TF-IDF shape:',X.shape)
K=range(2,min(8,len(df)-1)+1); scores=[]
for k in K:
 m=KMeans(n_clusters=k,random_state=42,n_init=10); y=m.fit_predict(X); scores.append(silhouette_score(X,y))
best_k=list(K)[int(np.argmax(scores))]; print('Optimal K:',best_k)
plt.figure(figsize=(7,4)); plt.plot(list(K),scores,marker='o'); plt.xlabel('K'); plt.ylabel('Silhouette Score'); plt.title('Optimal K'); plt.tight_layout(); plt.savefig('01_optimal_k.png',dpi=150); plt.show()
m=KMeans(n_clusters=best_k,random_state=42,n_init=10); df['KMeansCluster']=m.fit_predict(X)
svd=TruncatedSVD(n_components=2,random_state=42); X2=svd.fit_transform(X)
plt.figure(figsize=(8,6)); plt.scatter(X2[:,0],X2[:,1],c=df.KMeansCluster,cmap='tab10',s=60); plt.xlabel('SVD 1'); plt.ylabel('SVD 2'); plt.title('News Article Clusters - K-Means'); plt.tight_layout(); plt.savefig('02_kmeans_news_clusters.png',dpi=150); plt.show()
Z=linkage(X.toarray(),method='ward'); plt.figure(figsize=(10,5)); dendrogram(Z,truncate_mode='lastp',p=25); plt.title('Hierarchical Clustering Dendrogram'); plt.tight_layout(); plt.savefig('03_news_dendrogram.png',dpi=150); plt.show()
h=fcluster(Z,t=best_k,criterion='maxclust')-1; df['HierarchicalCluster']=h
print('K-Means Silhouette:',round(silhouette_score(X,df.KMeansCluster),4)); print('Hierarchical Silhouette:',round(silhouette_score(X,h),4))
df.to_csv('news_clustering_results.csv',index=False)
terms=vec.get_feature_names_out(); order=m.cluster_centers_.argsort()[:,::-1]; pd.DataFrame([{'Cluster':i,'TopTerms':', '.join(terms[j] for j in order[i,:10])} for i in range(best_k)]).to_csv('cluster_top_terms.csv',index=False)
