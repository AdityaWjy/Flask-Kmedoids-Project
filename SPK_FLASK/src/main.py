from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import io

app = Flask(__name__)

# Load dataset
file_path = 'Dataset Produksi Perikanan.xlsx'
df = pd.read_excel(file_path, sheet_name='rekapitulasi-produksi-perikanan')
data = df.drop(columns=['Jenis Ikan'])
data_np = data.values

# Variabel global
current_best_result = None
current_df_result = None
current_data_2d = None
results = []

def get_unique_medoid_indices(df, k, used_ikan):
    medoid_indices = []
    while len(medoid_indices) < k:
        idx = random.randint(0, len(df) - 1)
        jenis_ikan = df.iloc[idx]['Jenis Ikan']
        if jenis_ikan not in used_ikan:
            used_ikan.add(jenis_ikan)
            medoid_indices.append(idx)
        if len(used_ikan) == len(df['Jenis Ikan'].unique()):
            break
    if len(medoid_indices) < k:
        raise ValueError("Tidak bisa menemukan medoid unik yang cukup.")
    return medoid_indices

def kmedoids_once(X, k, used_ikan_global, max_iterations=100):
    medoid_indices = get_unique_medoid_indices(df, k, used_ikan_global.copy())
    current_medoids = np.array(medoid_indices)
    used_ikan = set(df.iloc[i]['Jenis Ikan'] for i in current_medoids)

    def compute_cost_and_labels(medoids):
        distances = pairwise_distances(X, X[medoids])
        labels = np.argmin(distances, axis=1)
        cost = np.sum(np.min(distances, axis=1))
        return cost, labels

    best_cost, best_labels = compute_cost_and_labels(current_medoids)
    best_medoids = current_medoids.copy()
    iterations = 0

    initial_cost = best_cost
    initial_medoids = best_medoids.copy()

    while iterations < max_iterations:
        iterations += 1

        try:
            new_indices = get_unique_medoid_indices(df, k, used_ikan.copy())
        except ValueError:
            break

        new_medoids = np.array(new_indices)
        new_cost, new_labels = compute_cost_and_labels(new_medoids)

        if new_cost < best_cost:
            best_cost = new_cost
            best_labels = new_labels
            best_medoids = new_medoids
            used_ikan.update(df.iloc[i]['Jenis Ikan'] for i in new_medoids)
        else:
            break

    return best_labels, best_medoids, best_cost, iterations, initial_medoids, initial_cost

@app.route('/')
def index():
    global current_best_result, current_df_result, current_data_2d, results
    results = []
    used_ikan = set()

    for _ in range(10):
        try:
            labels, medoids, cost, iterations, init_medoids, init_cost = kmedoids_once(data_np, 3, used_ikan)
            results.append({
                'labels': labels,
                'medoids': medoids,
                'cost': cost,
                'iterations': iterations,
                'init_medoids': init_medoids,
                'init_cost': init_cost
            })
        except ValueError:
            continue

    if not results:
        return "Gagal melakukan clustering karena tidak cukup medoid unik.", 500

    best_result = min(results, key=lambda x: x['cost'])
    best_index = [r['cost'] for r in results].index(best_result['cost'])

    df_result = df.copy()
    df_result['Cluster'] = best_result['labels']
    cluster_labels = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
    df_result['Hasil Cluster'] = df_result['Cluster'].map(cluster_labels)

    cols = list(df_result.columns)
    cols.remove('Hasil Cluster')
    cluster_index = cols.index('Cluster')
    cols.insert(cluster_index + 1, 'Hasil Cluster')
    df_result = df_result[cols]

    cluster_counts = df_result['Cluster'].value_counts().sort_index()
    cluster_counts_dict = {str(k): v for k, v in cluster_counts.items()}

    def get_medoid_names(indices):
        return [df.iloc[i]['Jenis Ikan'] for i in indices]

    all_medoid_sets = [get_medoid_names(r['medoids']) for r in results]
    all_init_medoid_sets = [get_medoid_names(r['init_medoids']) for r in results]
    all_costs = [r['cost'] for r in results]
    all_init_costs = [r['init_cost'] for r in results]
    all_iterations = [r['iterations'] for r in results]

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_np)

    current_best_result = best_result
    current_df_result = df_result
    current_data_2d = data_2d

    return render_template('index.html',
                           original_data=df.to_html(classes='table table-striped table-bordered text-start', index=False),
                           clustered_data=df_result.to_html(classes='table table-striped table-bordered text-start', index=False),
                           medoid_sets=all_medoid_sets,
                           init_medoid_sets=all_init_medoid_sets,
                           costs=all_costs,
                           init_costs=all_init_costs,
                           iterations=all_iterations,
                           best_index=best_index,
                           cluster_counts=cluster_counts_dict)

@app.route('/download')
def download():
    if current_df_result is None:
        return "Belum ada hasil clustering", 400

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        current_df_result.to_excel(writer, index=False, sheet_name='Hasil Cluster')
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='hasil_cluster.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

if __name__ == '__main__':
    app.run(debug=True)
