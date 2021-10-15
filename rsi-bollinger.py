def rsi(Data, rsi_lookback, which = 1): #OHLC Data
    
    rsi_lookback = (rsi_lookback * 2) - 1 # From exp to smoothed
    # Get the difference in price from previous step
    Data = pd.DataFrame(Data)
    delta = Data.iloc[:, 3].diff()
    delta = delta[1:] 
    
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = pd.stats.moments.ewma(up, rsi_lookback)
    roll_down = pd.stats.moments.ewma(down.abs(), rsi_lookback)
    
    roll_up = roll_up[rsi_lookback:]
    roll_down = roll_down[rsi_lookback:]
    Data = Data.iloc[rsi_lookback + 1:,].values
    
    # Calculate the RSI
    RS = roll_up / roll_down
    RSI = (100.0 - (100.0 / (1.0 + RS)))
    RSI = np.array(RSI)
    RSI = np.reshape(RSI, (-1, 1))
    
    Data = np.concatenate((Data, RSI), axis = 1)


def BollingerBands(Data, boll_lookback, volatility, onwhat, where_ma, where_up, where_down):
       
    # Calculating means
    for i in range(len(Data)):
        try:
            Data[i, where_ma] = (Data[i - boll_lookback:i + 1, onwhat].mean())
    
        except IndexError:
            pass
    for i in range(len(Data)):
            Data[i, where_up] = ((Data[i - boll_lookback:i, onwhat].std()) * volatility) + Data[i, where_ma]
            
    for i in range(len(Data)):
        Data[i, where_down] = Data[i, where_ma] - ((Data[i - boll_lookback:i, onwhat].std()) * volatility) 
            
    return Data