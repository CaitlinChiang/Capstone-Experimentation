from typing import List, Dict
from difflib import SequenceMatcher

def calculate_wer(prediction: str, truth: str) -> float:
    """
    Calculate the Word Error Rate (WER) between the predicted transcription and the truth.
    WER = (Substitutions + Insertions + Deletions) / Number of Words in Ground Truth
    """
    prediction_words = prediction.split()
    truth_words = truth.split()
    matcher = SequenceMatcher(None, truth_words, prediction_words)
    edit_operations = sum(op[0] != 'equal' for op in matcher.get_opcodes())
    return edit_operations / len(truth_words) if truth_words else 0

def calculate_word_metrics(prediction: str, truth: str) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score for the words in the transcription.
    """
    truth_words = truth.split()
    prediction_words = prediction.split()

    # Calculate true positives, false positives, and false negatives
    true_positives = len(set(prediction_words) & set(truth_words))
    total_predicted = len(prediction_words)
    total_actual = len(truth_words)

    # Precision, Recall, and F1
    precision = true_positives / total_predicted if total_predicted > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1}

def calculate_accuracy(prediction: str, truth: str) -> float:
    """
    Calculate the accuracy as the proportion of exact matches between prediction and truth.
    """
    return 1 if prediction == truth else 0

def evaluate_model(dataset: List[Dict], model_predictions: List[str]) -> Dict[str, float]:
    """
    Evaluate the model based on WER, Precision, Recall, F1-Score, and Accuracy.
    
    :param dataset: List of dictionaries with "truth" keys for ground truth sentences.
    :param model_predictions: List of strings containing the model's selected best transcriptions.
    """
    total_wer = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_accuracy = 0
    n = len(dataset)
    
    for i, data in enumerate(dataset):
        truth = data["truth"]
        prediction = model_predictions[i]
        
        total_wer += calculate_wer(prediction, truth)
        metrics = calculate_word_metrics(prediction, truth)
        total_precision += metrics["precision"]
        total_recall += metrics["recall"]
        total_f1 += metrics["f1_score"]
        total_accuracy += calculate_accuracy(prediction, truth)
    
    return {
        "average_wer": total_wer / n,
        "average_precision": total_precision / n,
        "average_recall": total_recall / n,
        "average_f1_score": total_f1 / n,
        "average_accuracy": total_accuracy / n
    }
