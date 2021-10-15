import pytest
from ..market_db import Database
from ..check_indicators import CheckIndicators

def test_get_common_mess():
    db = Database()
    chi = CheckIndicators()
    data = db.load_data(time_from="-10d", symbols=["INTC"])
    data0 = data.copy()
 
    # data.iloc[-1, data.columns.get_loc("close")] = 50
    data.iloc[-1, data.columns.get_loc("close")] = 50
    data.iloc[-2, data.columns.get_loc("close")] = 48
    data.iloc[-3, data.columns.get_loc("close")] = 45
    data.iloc[-4, data.columns.get_loc("close")] = 41
    data.iloc[-5, data.columns.get_loc("close")] = 55


    candles = chi.check_candles_in_row(data)
    
    print(candles)
    assert candles[1] == 3, " get_common_mess failed"
    
    data0.iloc[-1, data0.columns.get_loc("close")] = 40
    data0.iloc[-2, data0.columns.get_loc("close")] = 41
    data0.iloc[-3, data0.columns.get_loc("close")] = 42
    data0.iloc[-4, data0.columns.get_loc("close")] = 43
    data0.iloc[-5, data0.columns.get_loc("close")] = 39
    print("dataaaaaaaaaaaa0")
    print(data0)
    
    candles = chi.check_candles_in_row(data0)
    
    print(candles)

    assert candles[0] == 3, " get_common_mess failed"


def test_check_star():
    db = Database()
    chi = CheckIndicators()
    data = db.load_data(time_from="-10d", symbols=["INTC"])
    data0 = data.copy()
    
    # arrange star candles pattern two sequence of down candles and morning star after them indicator for long
    data.iloc[-1, data.columns.get_loc("close")] = 40
    data.iloc[-1, data.columns.get_loc("open")] = 42
    data.iloc[-1, data.columns.get_loc("high")] = 46
    data.iloc[-1, data.columns.get_loc("low")] = 36
  
    data.iloc[-2, data.columns.get_loc("close")] = 45
    data.iloc[-3, data.columns.get_loc("close")] = 50

    assert chi.check_star(
        data, buy=True) is True, " star pattern checker is wrong for buy option"

    assert chi.check_star(
        data, buy=False) is False, " star pattern checker is wrong for sell option"

    # arrange star candles pattern two sequence of up candles and morning star after them indicator for short
    data.iloc[-1, data.columns.get_loc("close")] = 45
    data.iloc[-1, data.columns.get_loc("open")] = 43
    data.iloc[-1, data.columns.get_loc("high")] = 49
    data.iloc[-1, data.columns.get_loc("low")] = 39

    data.iloc[-2, data.columns.get_loc("close")] = 40
    data.iloc[-3, data.columns.get_loc("close")] = 35

    assert chi.check_star(
        data, buy=True) is False, " star pattern checker is wrong for buy option"

    assert chi.check_star(
        data, buy=False) is True, " star pattern checker is wrong for sell option"


def test_check_hammer():
    db = Database()
    chi = CheckIndicators()
    data = db.load_data(time_from="-10d", symbols=["INTC"])
    data0 = data.copy()

    # arrange star candles pattern two sequence of down candles and morning star after them indicator for long
    data.iloc[-1, data.columns.get_loc("close")] = 40
    data.iloc[-1, data.columns.get_loc("open")] = 42
    data.iloc[-1, data.columns.get_loc("high")] = 43
    data.iloc[-1, data.columns.get_loc("low")] = 36

    data.iloc[-2, data.columns.get_loc("close")] = 45
    data.iloc[-3, data.columns.get_loc("close")] = 50

    assert chi.check_hammer(
        data, buy=True) is True, " star pattern checker is wrong for buy option"

    assert chi.check_hammer(
        data, buy=False) is False, " star pattern checker is wrong for sell option"

    # arrange star candles pattern two sequence of up candles and morning star after them indicator for short
    data.iloc[-1, data.columns.get_loc("close")] = 45
    data.iloc[-1, data.columns.get_loc("open")] = 43
    data.iloc[-1, data.columns.get_loc("high")] = 49
    data.iloc[-1, data.columns.get_loc("low")] = 44

    data.iloc[-2, data.columns.get_loc("close")] = 40
    data.iloc[-3, data.columns.get_loc("close")] = 35

    assert chi.check_hammer(
        data, buy=True) is False, " star pattern checker is wrong for buy option"

    assert chi.check_hammer(
        data, buy=False) is True, " star pattern checker is wrong for sell option"
