from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn import datasets

app = Flask(__name__)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvectors, eigenvalues = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[:self.n_components]
        self.explained_variance = eigenvalues[idxs[:self.n_components]] / np.sum(eigenvalues)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

@app.route("/", methods=["GET", "POST"])
def index():
    # Load Iris dataset
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Set number of components (default 2)
    n_components = 2
    if request.method == "POST":
        n_components = int(request.form.get("n_components", 2))

    # Apply PCA
    pca = PCA(n_components)
    pca.fit(X)
    X_projected = pca.transform(X)

    # Calculate explained variance (precision) as the cumulative variance explained by the components
    explained_variance = np.sum(pca.explained_variance)

    # Plotting the PCA results
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    fig, ax = plt.subplots()
    scatter = ax.scatter(x1, x2, c=y, cmap=plt.cm.viridis, edgecolor="none", alpha=0.8)
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    fig.colorbar(scatter)

    # Save plot to a string buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template("index.html", plot_url=plot_url, n_components=n_components, explained_variance=explained_variance)

if __name__ == "__main__":
    app.run(debug=True)
