

class Indicators():
    
    @staticmethod
    def SMA(df, periods=50):
        """
        Calculating the Simple Moving Average for the past n days
        
        **Values must be descending**
        """
        lst = []
            
        for i in range(len(df)):
            if i < periods:
                
                # Appending NaNs for instances unable to look back on
                lst.append(np.nan)
                
            else:
                # Calculating the SMA
                lst.append(round(np.mean(df[i:periods+i]), 2))
            
        return lst
  
  
    @staticmethod
    def Stoch(closes, lows, highs, periods=14, d_periods=3):
        """
        Calculating the Stochastic Oscillator for the past n days
        
        **Values must be descending**
        """
        
        
        k_lst = []
        
        d_lst = []
        
        for i in range(len(closes)):
            if i < periods:
                
                # Appending NaNs for instances unable to look back on
                k_lst.append(np.nan)
                
                d_lst.append(np.nan)
                
            else:
                
                # Calculating the Stochastic Oscillator
                
                # Calculating the %K line
                highest = max(highs[i:periods+i])
                lowest = min(lows[i:periods+i])
                
                k = ((closes[i] - lowest) / (highest - lowest)) * 100
                
                k_lst.append(round(k, 2))
                
                # Calculating the %D line
                if len(k_lst) < d_periods:
                    d_lst.append(np.nan)
                else:
                    d_lst.append(round(np.mean(k_lst[-d_periods-1:-1])))
        
        return k_lst, d_lst
        
    @staticmethod    
    def RSI(df, periods=14):
        """
        Calculates the Relative Strength Index
        
        **Values must be descending**
        """
        
        df = df.diff()
        
        lst = []
        
        for i in range(len(df)):
            if i < periods:
                
                # Appending NaNs for instances unable to look back on
                lst.append(np.nan)
                
            else:
                
                # Calculating the Relative Strength Index          
                avg_gain = (sum([x for x in df[i:periods+i] if x >= 0]) / periods)
                avg_loss = (sum([abs(x) for x in df[i:periods+i] if x <= 0]) / periods)


                rs = avg_gain / avg_loss

                rsi = 100 - (100 / (1 + rs))

                lst.append(round(rsi, 2))

                
        return lst