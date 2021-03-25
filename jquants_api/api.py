import json
import requests
import pandas as pd


def call_refresh_api(refresh_token: str) -> dict:
    """
    Refresh IdToken

    Parameters
    ----------
    refresh_token : str
        refresh token.

    Returns
    -------
    out : dict

    """
    headers = {"accept": "application/json"}
    data = {"refresh-token": refresh_token}

    response = requests.post(
        "https://api.jpx-jquants.com/refresh",
        headers=headers,
        data=json.dumps(data),
    )

    out = json.loads(response.text)
    return out


def call_jquants_api(
    params: dict, id_token: str, api_type: str, code: str = None
) -> dict:
    """
    Call J-Quants to get stock lists, prices, labels etc.

    Parameters
    ----------
    params : dict
        Parameters dictionary.
    id_token : str
        idToken.
    api_type: str
        API type -> "stockfins", prices", "lists", and "stocklabels".
    code: str
        stock code.

    Returns
    -------
    out : dict

    """
    date_from = params.get("date_from", None)
    date_to = params.get("date_to", None)
    date = params.get("date", None)
    include_details = params.get("include_details", "false")
    keyword = params.get("keyword", None)
    headline = params.get("headline", None)
    param_code = params.get("code", None)
    next_token = params.get("next_token", None)
    headers = {"accept": "application/json", "Authorization": id_token}
    data = {
        "from": date_from,
        "to": date_to,
        "includeDetails": include_details,
        "nexttoken": next_token,
        "date": date,
        "keyword": keyword,
        "headline": headline,
        "code": param_code,
    }

    if code:
        code = "/" + code
        r = requests.get(
            "https://api.jpx-jquants.com/" + api_type + code,
            params=data,
            headers=headers,
        )
    else:
        r = requests.get(
            "https://api.jpx-jquants.com/" + api_type,
            params=data,
            headers=headers,
        )
    out = json.loads(r.text)
    return out


def get_labels(start_date: str, end_date: str, id_token: str) -> pd.DataFrame:
    """
    Get labels for a given data range
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        id_token (str): ID token

    Returns:
        pd.DataFrame: A dataframe with labels same as stock_labels data

    """
    df = pd.DataFrame(
        columns=[
            "label_low_10",
            "label_low_20",
            "label_low_5",
            "base_date",
            "label_high_20",
            "label_high_10",
            "label_date_5",
            "label_date_10",
            "label_high_5",
            "label_date_20",
            "Local Code",
        ]
    )
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")

    for date in dates:
        print(date)
        param_dict = {"date": date, "include_details": "true"}
        out = call_jquants_api(param_dict, id_token, "stocklabels")
        if len(out["labels"]) > 0:
            df = df.append(pd.DataFrame(out["labels"]))

    df = df[
        [
            "base_date",
            "Local Code",
            "label_date_5",
            "label_high_5",
            "label_low_5",
            "label_date_10",
            "label_high_10",
            "label_low_10",
            "label_date_20",
            "label_high_20",
            "label_low_20",
        ]
    ]
    df.reset_index(drop=True, inplace=True)
    return df


def get_prices(start_date: str, end_date: str, id_token: str) -> pd.DataFrame:
    """
    Get Prices for a given data range.
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        id_token (str): ID token

    Returns:
        pd.DataFrame: A dataframe with labels same as stock_prices data

    """
    df = pd.DataFrame(
        columns=[
            "EndOfDayQuote Open",
            "EndOfDayQuote PreviousClose",
            "EndOfDayQuote CumulativeAdjustmentFactor",
            "EndOfDayQuote VWAP",
            "EndOfDayQuote Low",
            "EndOfDayQuote PreviousExchangeOfficialClose",
            "EndOfDayQuote High",
            "EndOfDayQuote Date",
            "EndOfDayQuote Close",
            "EndOfDayQuote PreviousExchangeOfficialCloseDate",
            "EndOfDayQuote ExchangeOfficialClose",
            "EndOfDayQuote ChangeFromPreviousClose",
            "EndOfDayQuote PercentChangeFromPreviousClose",
            "EndOfDayQuote PreviousCloseDate",
            "Local Code",
            "EndOfDayQuote Volume",
        ]
    )
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d")

    for date in dates:
        print(date)
        param_dict = {"date": date, "include_details": "true"}
        out = call_jquants_api(param_dict, id_token, "prices")
        if len(out["prices"]) > 0:
            df = df.append(pd.DataFrame(out["prices"]))

    df = df[
        [
            "Local Code",
            "EndOfDayQuote Date",
            "EndOfDayQuote Open",
            "EndOfDayQuote High",
            "EndOfDayQuote Low",
            "EndOfDayQuote Close",
            "EndOfDayQuote ExchangeOfficialClose",
            "EndOfDayQuote Volume",
            "EndOfDayQuote CumulativeAdjustmentFactor",
            "EndOfDayQuote PreviousClose",
            "EndOfDayQuote PreviousCloseDate",
            "EndOfDayQuote PreviousExchangeOfficialClose",
            "EndOfDayQuote PreviousExchangeOfficialCloseDate",
            "EndOfDayQuote ChangeFromPreviousClose",
            "EndOfDayQuote PercentChangeFromPreviousClose",
            "EndOfDayQuote VWAP",
        ]
    ]
    df.reset_index(drop=True, inplace=True)
    return df
