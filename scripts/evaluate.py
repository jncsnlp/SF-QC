#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import classification_report


LABEL_SETS: dict[str, list[str]] = {'banking10': ['pending_cash_withdrawal',
            'balance_not_updated_after_cheque_or_cash_deposit',
            'balance_not_updated_after_bank_transfer',
            'top_up_limits',
            'direct_debit_payment_not_recognised',
            'declined_card_payment',
            'beneficiary_not_allowed'],
 'banking25': ['pending_cash_withdrawal',
            'balance_not_updated_after_cheque_or_cash_deposit',
            'balance_not_updated_after_bank_transfer',
            'top_up_limits',
            'direct_debit_payment_not_recognised',
            'declined_card_payment',
            'beneficiary_not_allowed',
            'supported_cards_and_currencies',
            'pending_transfer',
            'card_linking',
            'virtual_card_not_working',
            'extra_charge_on_statement',
            'apple_pay_or_google_pay',
            'card_about_to_expire',
            'getting_spare_card',
            'wrong_amount_of_cash_received',
            'cancel_transfer',
            'unable_to_verify_identity',
            'fiat_currency_support'],
 'banking25sim': ['card_not_working',
               'declined_card_payment',
               'card_payment_not_recognised',
               'pending_card_payment',
               'reverted_card_payment?',
               'card_payment_fee_charged',
               'card_payment_wrong_exchange_rate',
               'extra_charge_on_statement',
               'transaction_charged_twice',
               'contactless_not_working',
               'virtual_card_not_working',
               'getting_virtual_card',
               'get_disposable_virtual_card',
               'disposable_card_limits',
               'activate_my_card',
               'change_pin',
               'pin_blocked',
               'lost_or_stolen_card',
               'compromised_card'],
 'banking50': ['pending_cash_withdrawal',
            'balance_not_updated_after_cheque_or_cash_deposit',
            'balance_not_updated_after_bank_transfer',
            'top_up_limits',
            'direct_debit_payment_not_recognised',
            'declined_card_payment',
            'beneficiary_not_allowed',
            'supported_cards_and_currencies',
            'pending_transfer',
            'card_linking',
            'virtual_card_not_working',
            'extra_charge_on_statement',
            'apple_pay_or_google_pay',
            'card_about_to_expire',
            'getting_spare_card',
            'wrong_amount_of_cash_received',
            'cancel_transfer',
            'unable_to_verify_identity',
            'fiat_currency_support',
            'card_payment_wrong_exchange_rate',
            'card_payment_not_recognised',
            'topping_up_by_card',
            'verify_my_identity',
            'getting_virtual_card',
            'wrong_exchange_rate_for_cash_withdrawal',
            'activate_my_card',
            'lost_or_stolen_card',
            'automatic_top_up',
            'transfer_not_received_by_recipient',
            'declined_transfer',
            'compromised_card',
            'transaction_charged_twice',
            'Refund_not_showing_up',
            'top_up_by_card_charge',
            'declined_cash_withdrawal',
            'transfer_fee_charged',
            'get_disposable_virtual_card',
            'card_delivery_estimate'],
 'banking75': ['pending_cash_withdrawal',
            'balance_not_updated_after_cheque_or_cash_deposit',
            'balance_not_updated_after_bank_transfer',
            'top_up_limits',
            'direct_debit_payment_not_recognised',
            'declined_card_payment',
            'beneficiary_not_allowed',
            'supported_cards_and_currencies',
            'pending_transfer',
            'card_linking',
            'virtual_card_not_working',
            'extra_charge_on_statement',
            'apple_pay_or_google_pay',
            'card_about_to_expire',
            'getting_spare_card',
            'wrong_amount_of_cash_received',
            'cancel_transfer',
            'unable_to_verify_identity',
            'fiat_currency_support',
            'card_payment_wrong_exchange_rate',
            'card_payment_not_recognised',
            'topping_up_by_card',
            'verify_my_identity',
            'getting_virtual_card',
            'wrong_exchange_rate_for_cash_withdrawal',
            'activate_my_card',
            'lost_or_stolen_card',
            'automatic_top_up',
            'transfer_not_received_by_recipient',
            'declined_transfer',
            'compromised_card',
            'transaction_charged_twice',
            'Refund_not_showing_up',
            'top_up_by_card_charge',
            'declined_cash_withdrawal',
            'transfer_fee_charged',
            'get_disposable_virtual_card',
            'card_delivery_estimate',
            'order_physical_card',
            'cash_withdrawal_not_recognised',
            'top_up_by_bank_transfer_charge',
            'card_not_working',
            'transfer_timing',
            'visa_or_mastercard',
            'card_acceptance',
            'card_swallowed',
            'atm_support',
            'passcode_forgotten',
            'exchange_charge',
            'pending_top_up',
            'terminate_account',
            'age_limit',
            'top_up_reverted',
            'top_up_by_cash_or_cheque',
            'lost_or_stolen_phone',
            'exchange_via_app',
            'verify_source_of_funds'],
 'banking_all': ['exchange_charge',
              'balance_not_updated_after_cheque_or_cash_deposit',
              'beneficiary_not_allowed',
              'pending_card_payment',
              'top_up_limits',
              'get_disposable_virtual_card',
              'pending_top_up',
              'supported_cards_and_currencies',
              'transfer_into_account',
              'verify_source_of_funds',
              'top_up_failed',
              'verify_top_up',
              'why_verify_identity',
              'verify_my_identity',
              'extra_charge_on_statement',
              'card_arrival',
              'fiat_currency_support',
              'cancel_transfer',
              'pending_transfer',
              'automatic_top_up',
              'direct_debit_payment_not_recognised',
              'failed_transfer',
              'wrong_amount_of_cash_received',
              'cash_withdrawal_not_recognised',
              'edit_personal_details',
              'unable_to_verify_identity',
              'pending_cash_withdrawal',
              'change_pin',
              'top_up_by_bank_transfer_charge',
              'top_up_by_card_charge',
              'declined_card_payment',
              'transfer_fee_charged',
              'declined_transfer',
              'disposable_card_limits',
              'exchange_via_app',
              'card_payment_fee_charged',
              'Refund_not_showing_up',
              'reverted_card_payment?',
              'transfer_timing',
              'country_support',
              'top_up_reverted',
              'card_swallowed',
              'passcode_forgotten',
              'apple_pay_or_google_pay',
              'order_physical_card',
              'getting_spare_card',
              'transfer_not_received_by_recipient',
              'card_linking',
              'declined_cash_withdrawal',
              'top_up_by_cash_or_cheque',
              'transaction_charged_twice',
              'wrong_exchange_rate_for_cash_withdrawal',
              'card_not_working',
              'lost_or_stolen_phone',
              'topping_up_by_card',
              'request_refund',
              'lost_or_stolen_card',
              'getting_virtual_card',
              'contactless_not_working',
              'card_about_to_expire',
              'age_limit',
              'card_payment_wrong_exchange_rate',
              'exchange_rate',
              'visa_or_mastercard',
              'card_delivery_estimate',
              'card_payment_not_recognised',
              'balance_not_updated_after_bank_transfer',
              'terminate_account',
              'get_physical_card',
              'pin_blocked',
              'receiving_money',
              'virtual_card_not_working',
              'activate_my_card',
              'card_acceptance',
              'atm_support',
              'cash_withdrawal_charge',
              'compromised_card'],
 'clinc25': ['no',
             'timer',
             'are_you_a_bot',
             'improve_credit_score',
             'credit_score',
             'pto_used',
             'meaning_of_life',
             'time',
             'pay_bill',
             'order_status',
             'recipe',
             'who_made_you',
             'tire_change',
             'reset_settings',
             'credit_limit',
             'ingredient_substitution',
             'last_maintenance',
             'food_last',
             'next_holiday',
             'calculator',
             'tire_pressure',
             'who_do_you_work_for',
             'restaurant_reviews',
             'rollover_401k',
             'pto_request_status',
             'greeting',
             'accept_reservations',
             'thank_you',
             'card_declined',
             'tell_joke',
             'alarm',
             'redeem_rewards',
             'current_location',
             'direct_deposit',
             'schedule_maintenance',
             'fun_fact',
             'date'],
 'clinc50': ['no',
             'timer',
             'are_you_a_bot',
             'improve_credit_score',
             'credit_score',
             'pto_used',
             'meaning_of_life',
             'time',
             'pay_bill',
             'order_status',
             'recipe',
             'who_made_you',
             'tire_change',
             'reset_settings',
             'credit_limit',
             'ingredient_substitution',
             'last_maintenance',
             'food_last',
             'next_holiday',
             'calculator',
             'tire_pressure',
             'who_do_you_work_for',
             'restaurant_reviews',
             'rollover_401k',
             'pto_request_status',
             'greeting',
             'accept_reservations',
             'thank_you',
             'card_declined',
             'tell_joke',
             'alarm',
             'redeem_rewards',
             'current_location',
             'direct_deposit',
             'schedule_maintenance',
             'fun_fact',
             'date',
             'make_call',
             'ingredients_list',
             'yes',
             'calendar',
             'cancel',
             'income',
             'play_music',
             'do_you_have_pets',
             'uber',
             'change_speed',
             'international_fees',
             'pto_request',
             'pto_balance',
             'order_checks',
             'balance',
             'flip_coin',
             'cook_time',
             'maybe',
             'calories',
             'book_hotel',
             'how_busy',
             'whisper_mode',
             'rewards_balance',
             'calendar_update',
             'travel_notification',
             'where_are_you_from',
             'payday',
             'jump_start',
             'reminder',
             'find_phone',
             'order',
             'international_visa',
             'gas',
             'car_rental',
             'report_fraud',
             'sync_device',
             'what_is_your_name',
             'pin_change'],
 'clinc75': ['no',
             'timer',
             'are_you_a_bot',
             'improve_credit_score',
             'credit_score',
             'pto_used',
             'meaning_of_life',
             'time',
             'pay_bill',
             'order_status',
             'recipe',
             'who_made_you',
             'tire_change',
             'reset_settings',
             'credit_limit',
             'ingredient_substitution',
             'last_maintenance',
             'food_last',
             'next_holiday',
             'calculator',
             'tire_pressure',
             'who_do_you_work_for',
             'restaurant_reviews',
             'rollover_401k',
             'pto_request_status',
             'greeting',
             'accept_reservations',
             'thank_you',
             'card_declined',
             'tell_joke',
             'alarm',
             'redeem_rewards',
             'current_location',
             'direct_deposit',
             'schedule_maintenance',
             'fun_fact',
             'date',
             'make_call',
             'ingredients_list',
             'yes',
             'calendar',
             'cancel',
             'income',
             'play_music',
             'do_you_have_pets',
             'uber',
             'change_speed',
             'international_fees',
             'pto_request',
             'pto_balance',
             'order_checks',
             'balance',
             'flip_coin',
             'cook_time',
             'maybe',
             'calories',
             'book_hotel',
             'how_busy',
             'whisper_mode',
             'rewards_balance',
             'calendar_update',
             'travel_notification',
             'where_are_you_from',
             'payday',
             'jump_start',
             'reminder',
             'find_phone',
             'order',
             'international_visa',
             'gas',
             'car_rental',
             'report_fraud',
             'sync_device',
             'what_is_your_name',
             'pin_change',
             'reminder_update',
             'bill_due',
             'oil_change_when',
             'vaccines',
             'distance',
             'book_flight',
             'todo_list',
             'what_can_i_ask_you',
             'spelling',
             'todo_list_update',
             'carry_on',
             'repeat',
             'nutrition_info',
             'shopping_list',
             'gas_type',
             'meal_suggestion',
             'insurance',
             'directions',
             'update_playlist',
             'account_blocked',
             'report_lost_card',
             'smart_home',
             'travel_alert',
             'credit_limit_change',
             'user_name',
             'restaurant_reservation',
             'interest_rate',
             'translate',
             'flight_status',
             'insurance_change',
             'change_user_name',
             'routing',
             'expiration_date',
             'change_ai_name',
             'definition',
             'min_payment',
             'text'],
 'clinc_all': ['translate',
               'transfer',
               'timer',
               'definition',
               'meaning_of_life',
               'insurance_change',
               'find_phone',
               'travel_alert',
               'pto_request',
               'improve_credit_score',
               'fun_fact',
               'change_language',
               'payday',
               'replacement_card_duration',
               'time',
               'application_status',
               'flight_status',
               'flip_coin',
               'change_user_name',
               'where_are_you_from',
               'shopping_list_update',
               'what_can_i_ask_you',
               'maybe',
               'oil_change_how',
               'restaurant_reservation',
               'balance',
               'confirm_reservation',
               'freeze_account',
               'rollover_401k',
               'who_made_you',
               'distance',
               'user_name',
               'timezone',
               'next_song',
               'transactions',
               'restaurant_suggestion',
               'rewards_balance',
               'pay_bill',
               'spending_history',
               'pto_request_status',
               'credit_score',
               'new_card',
               'lost_luggage',
               'repeat',
               'mpg',
               'oil_change_when',
               'yes',
               'travel_suggestion',
               'insurance',
               'todo_list_update',
               'reminder',
               'change_speed',
               'tire_pressure',
               'no',
               'apr',
               'nutrition_info',
               'calendar',
               'uber',
               'calculator',
               'date',
               'carry_on',
               'pto_used',
               'schedule_maintenance',
               'travel_notification',
               'sync_device',
               'thank_you',
               'roll_dice',
               'food_last',
               'cook_time',
               'reminder_update',
               'report_lost_card',
               'ingredient_substitution',
               'make_call',
               'alarm',
               'todo_list',
               'change_accent',
               'w2',
               'bill_due',
               'calories',
               'damaged_card',
               'restaurant_reviews',
               'routing',
               'do_you_have_pets',
               'schedule_meeting',
               'gas_type',
               'plug_type',
               'tire_change',
               'exchange_rate',
               'next_holiday',
               'change_volume',
               'who_do_you_work_for',
               'credit_limit',
               'how_busy',
               'accept_reservations',
               'order_status',
               'pin_change',
               'goodbye',
               'account_blocked',
               'what_song',
               'international_fees',
               'last_maintenance',
               'meeting_schedule',
               'ingredients_list',
               'report_fraud',
               'measurement_conversion',
               'smart_home',
               'book_hotel',
               'current_location',
               'weather',
               'taxes',
               'min_payment',
               'whisper_mode',
               'cancel',
               'international_visa',
               'vaccines',
               'pto_balance',
               'directions',
               'spelling',
               'greeting',
               'reset_settings',
               'what_is_your_name',
               'direct_deposit',
               'interest_rate',
               'credit_limit_change',
               'what_are_your_hobbies',
               'book_flight',
               'shopping_list',
               'text',
               'bill_balance',
               'share_location',
               'redeem_rewards',
               'play_music',
               'calendar_update',
               'are_you_a_bot',
               'gas',
               'expiration_date',
               'update_playlist',
               'cancel_reservation',
               'tell_joke',
               'change_ai_name',
               'how_old_are_you',
               'car_rental',
               'jump_start',
               'meal_suggestion',
               'recipe',
               'income',
               'order',
               'traffic',
               'order_checks',
               'card_declined']}


@dataclass(frozen=True)
class EvaluationResult:
    """Save the core evaluation results."""

    label_set: str
    sample_count: int
    evaluated_sample_count: int
    id_f1: float
    id_accuracy: float
    ood_f1: float
    ood_accuracy: float | None
    all_f1: float
    all_accuracy: float
    per_class_accuracy: dict[str, float | None]
    classification_report: dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the performance of a model on a given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the predictions CSV file.",
    )
    parser.add_argument(
        "--type",
        choices=sorted(LABEL_SETS),
        default="clinc25",
        help=f"Label set for evaluation. You can choose from {', '.join(sorted(LABEL_SETS))}.",
    )
    parser.add_argument(
        "--true_column",
        default="label",
        help="CSV column name for true labels.",
    )
    parser.add_argument(
        "--pred_column",
        default="answer",
        help="CSV column name for predicted labels.",
    )
    parser.add_argument(
        "--unknown_label",
        default="unknown",
        help="OOD label for unknown intents.",
    )
    parser.add_argument(
        "--show_labels",
        action="store_true",
        help="Output unique true labels and predicted labels.",
    )
    parser.add_argument(
        "--show_report",
        action="store_true",
        help="Output full classification report.",
    )
    parser.add_argument(
        "--show_class_accuracy",
        action="store_true",
        help="Output accuracy per class.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Save evaluation results as JSON file.",
    )
    return parser.parse_args()


def load_predictions(
    input_file: Path,
    true_column: str,
    pred_column: str,
) -> tuple[pd.Series, pd.Series]:
    """Load predictions from CSV file and return true labels and predicted labels."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file {input_file} not found.")

    dataframe = pd.read_csv(input_file)
    required_columns = {true_column, pred_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing_text}")

    valid_rows = dataframe[[true_column, pred_column]].dropna()
    if valid_rows.empty:
        raise ValueError("CSV has no valid label data for evaluation.")

    true_labels = valid_rows[true_column].astype(str).str.strip()
    predicted_labels = valid_rows[pred_column].astype(str).str.strip()
    return true_labels, predicted_labels


def unique_preserving_order(labels: Sequence[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys(labels))


def calculate_per_class_accuracy(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    labels: Sequence[str],
) -> tuple[dict[str, float | None], int, int]:
    """Calculate accuracy per class, total correct samples, and total samples。"""
    accuracy_per_class: dict[str, float | None] = {}
    total_correct = 0
    total_samples = 0

    for label in labels:
        class_mask = true_labels.eq(label)
        class_total = int(class_mask.sum())
        class_correct = int(predicted_labels[class_mask].eq(label).sum())

        accuracy_per_class[label] = (
            class_correct / class_total if class_total > 0 else None
        )
        total_correct += class_correct
        total_samples += class_total

    return accuracy_per_class, total_correct, total_samples


def evaluate_predictions(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    id_labels: Sequence[str],
    unknown_label: str = "unknown",
    label_set_name: str = "custom",
) -> EvaluationResult:
    """Evaluate ID、OOD 和 overall metrics。"""
    normalized_id_labels = unique_preserving_order(id_labels)
    normalized_id_labels = [
        label for label in normalized_id_labels if label != unknown_label
    ]
    if not normalized_id_labels:
        raise ValueError("ID label set cannot be empty.")

    all_labels = [*normalized_id_labels, unknown_label]

    id_report = classification_report(
        true_labels,
        predicted_labels,
        labels=normalized_id_labels,
        output_dict=True,
        zero_division=0,
    )
    all_report = classification_report(
        true_labels,
        predicted_labels,
        labels=all_labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_accuracy, total_correct, evaluated_sample_count = (
        calculate_per_class_accuracy(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            labels=all_labels,
        )
    )
    if evaluated_sample_count == 0:
        raise ValueError("No samples found in true labels.")

    id_class_accuracies = [
        per_class_accuracy[label]
        for label in normalized_id_labels
        if per_class_accuracy[label] is not None
    ]
    if not id_class_accuracies:
        raise ValueError("ID label set not found in true labels.")

    return EvaluationResult(
        label_set=label_set_name,
        sample_count=len(true_labels),
        evaluated_sample_count=evaluated_sample_count,
        id_f1=float(id_report["weighted avg"]["f1-score"]),
        id_accuracy=float(sum(id_class_accuracies) / len(id_class_accuracies)),
        ood_f1=float(all_report[unknown_label]["f1-score"]),
        ood_accuracy=per_class_accuracy[unknown_label],
        all_f1=float(all_report["macro avg"]["f1-score"]),
        all_accuracy=float(total_correct / evaluated_sample_count),
        per_class_accuracy=per_class_accuracy,
        classification_report=all_report,
    )


def format_optional_metric(value: float | None) -> str:
    """Format optional metric value。"""
    return "N/A" if value is None else f"{value:.6f}"


def print_summary(result: EvaluationResult) -> None:
    """Print evaluation summary。"""
    print("\n========== Evaluation Summary ==========")
    print(f"Label set         : {result.label_set}")
    print(f"Input samples     : {result.sample_count}")
    print(f"Evaluated samples : {result.evaluated_sample_count}")
    print(
        "OOD F1 / ACC      : "
        f"{result.ood_f1:.6f} / {format_optional_metric(result.ood_accuracy)}"
    )
    print(f"ALL F1 / ACC      : {result.all_f1:.6f} / {result.all_accuracy:.6f}")
    print(f"ID F1 / ACC       : {result.id_f1:.6f} / {result.id_accuracy:.6f}")


def print_class_accuracy(per_class_accuracy: dict[str, float | None]) -> None:
    """Print accuracy per class。"""
    print("\n========== Accuracy Per Class ==========")
    for label, accuracy in per_class_accuracy.items():
        print(f"{label}: {format_optional_metric(accuracy)}")


def save_result(result: EvaluationResult, output_file: Path) -> None:
    """Save evaluation results to JSON file。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to {output_file}")


def run_evaluation(
    input_file: Path,
    label_set: str,
    true_column: str = "label",
    pred_column: str = "answer",
    unknown_label: str = "unknown",
    show_labels: bool = False,
    show_report: bool = False,
    show_class_accuracy: bool = False,
    output_json: Path | None = None,
) -> EvaluationResult:

    true_labels, predicted_labels = load_predictions(
        input_file=input_file,
        true_column=true_column,
        pred_column=pred_column,
    )

    if show_labels:
        print("Unique true labels:", sorted(true_labels.unique().tolist()))
        print("Unique predicted labels:", sorted(predicted_labels.unique().tolist()))

    id_labels = LABEL_SETS[label_set]
    result = evaluate_predictions(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        id_labels=id_labels,
        unknown_label=unknown_label,
        label_set_name=label_set,
    )

    if show_class_accuracy:
        print_class_accuracy(result.per_class_accuracy)

    if show_report:
        print("\n========== Classification Report ==========")
        report_labels = [
            *[
                label
                for label in unique_preserving_order(id_labels)
                if label != unknown_label
            ],
            unknown_label,
        ]
        print(
            classification_report(
                true_labels,
                predicted_labels,
                labels=report_labels,
                zero_division=0,
                digits=6,
            )
        )

    print_summary(result)
    if output_json is not None:
        save_result(result, output_json)

    return result


def main() -> None:
    args = parse_args()
    run_evaluation(
        input_file=args.input_file,
        label_set=args.type,
        true_column=args.true_column,
        pred_column=args.pred_column,
        unknown_label=args.unknown_label,
        show_labels=args.show_labels,
        show_report=args.show_report,
        show_class_accuracy=args.show_class_accuracy,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
