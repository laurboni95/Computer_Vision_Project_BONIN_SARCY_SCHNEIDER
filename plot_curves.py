import matplotlib.pyplot as plt
import numpy as np
import json

def plot_predictions():
    # Load predictions for 10 examples
    with open('predictions_10.json', 'r') as f:
        preds_mlp = json.load(f)
    with open('predictions_10_gru.json', 'r') as f:
        preds_gru = json.load(f)
    
    probs_mlp = [p['prob'] for p in preds_mlp]
    probs_gru = [p['prob'] for p in preds_gru]
    
    indices = list(range(len(probs_mlp)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(indices, probs_mlp, label='MLP', alpha=0.7)
    ax.bar(indices, probs_gru, label='GRU', alpha=0.7)
    ax.set_xlabel('Example Index')
    ax.set_ylabel('Probability')
    ax.set_title('Predictions for 10 Examples')
    ax.legend()
    plt.savefig('example_pngs/predictions_plot.png')
    plt.close()

    # Separate for GRU
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(indices, probs_gru, color='orange', alpha=0.7)
    ax.set_xlabel('Example Index')
    ax.set_ylabel('Probability')
    ax.set_title('GRU Predictions for 10 Examples')
    plt.savefig('example_pngs/predictions_plot_gru.png')
    plt.close()

def plot_scatter_sampled():
    with open('predictions_pie_sampled.json', 'r') as f:
        data = json.load(f)
    
    probs_mlp = [p['prob_mlp'] for p in data['predictions']]
    probs_gru = [p['prob_gru'] for p in data['predictions']]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(probs_mlp, probs_gru, alpha=0.5)
    ax.set_xlabel('MLP Probabilities')
    ax.set_ylabel('GRU Probabilities')
    ax.set_title('MLP vs GRU Probabilities Scatter Plot - PIE Sampled')
    plt.savefig('example_pngs/scatter_mlp_vs_gru_sampled.png')
    plt.close()

def plot_smoothed_scatter():
    with open('predictions_pie_sampled.json', 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    # Sort by sample_index
    predictions.sort(key=lambda x: x['sample_index'])
    
    indices = [p['sample_index'] for p in predictions]
    probs_mlp = [p['prob_mlp'] for p in predictions]
    probs_gru = [p['prob_gru'] for p in predictions]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original points
    ax.scatter(indices, probs_mlp, alpha=0.5, label='MLP Raw', color='blue')
    ax.scatter(indices, probs_gru, alpha=0.5, label='GRU Raw', color='orange')
    
    # Smooth curves
    z_mlp = np.polyfit(indices, probs_mlp, 5)
    p_mlp = np.poly1d(z_mlp)
    z_gru = np.polyfit(indices, probs_gru, 5)
    p_gru = np.poly1d(z_gru)
    
    x_smooth = np.linspace(min(indices), max(indices), 500)
    ax.plot(x_smooth, p_mlp(x_smooth), 'b-', linewidth=2, label='MLP Smoothed')
    ax.plot(x_smooth, p_gru(x_smooth), 'r-', linewidth=2, label='GRU Smoothed')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Probability')
    ax.set_title('Smoothed Probabilities vs Sample Index - PIE Dataset')
    ax.legend()
    plt.savefig('example_pngs/compare_pie_mlp_gru_smoothed_deux.png')
    plt.close()

def plot_failure_example():
    with open('predictions_10.json', 'r') as f:
        preds = json.load(f)
    
    # Assume sel1_s42 means selection 1, sample 42, perhaps index 42
    if len(preds) > 42:
        example = preds[42]
        bboxes = example['bboxes']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, str(i), fontsize=8, color='blue')
        
        ax.set_xlim(0, 1920)  # Assume image width
        ax.set_ylim(1080, 0)  # Assume image height, inverted
        ax.set_aspect('equal')
        ax.set_title('Failure Example - Bboxes for Sample 42')
        plt.savefig('example_pngs/failure_example_sel1_s42.png')
        plt.close()

def plot_metrics_comparison():
    # Load metrics
    with open('metrics_mlp_gru.json', 'r') as f:
        metrics_full = json.load(f)
    
    with open('metrics_pie_sampled.json', 'r') as f:
        metrics_pie = json.load(f)
    
    # For JAAD comparison
    models = ['MLP', 'GRU']
    accuracy = [metrics_full['mlp']['accuracy'], metrics_full['gru']['accuracy']]
    auc_scores = [metrics_full['mlp']['auc'], metrics_full['gru']['auc']]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracy, width, label='Accuracy')
    ax.bar(x + width/2, auc_scores, width, label='AUC')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('MLP vs GRU Metrics Comparison - JAAD')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.savefig('example_pngs/compare_mlp_gru.png')
    plt.close()
    
    # For PIE comparison
    accuracy_pie = [metrics_pie['acc_mlp'], metrics_pie['acc_gru']]
    auc_pie = [metrics_pie['auc_mlp'], metrics_pie['auc_gru']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, accuracy_pie, width, label='Accuracy')
    ax.bar(x + width/2, auc_pie, width, label='AUC')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('MLP vs GRU Metrics Comparison - PIE')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.savefig('example_pngs/compare_pie_mlp_gru.png')
    plt.close()

if __name__ == "__main__":
    plot_predictions()
    plot_scatter_sampled()
    plot_smoothed_scatter()
    plot_failure_example()
    plot_metrics_comparison()
    print("All plots generated and saved to example_pngs/")