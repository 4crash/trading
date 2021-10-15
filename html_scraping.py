from bs4 import BeautifulSoup
import requests
# import selenium.webdriver as webdriver
1
2
3


class HtmlScarpe(object):
    """
    docstring
    <p class="rank_view">
                       2-Buy<span class="sr-only"> of 5</span> <span class="rank_chip rankrect_1">&nbsp;</span> 
                       <span class="rank_chip rankrect_2">2</span> <span class="rank_chip rankrect_3">&nbsp;</span> 
                       <span class="rank_chip rankrect_4">&nbsp;</span> <span class="rank_chip rankrect_5">&nbsp;</span>                </p>
            </div>
    """
    def __init__(self):
        """
        docstring
        """
        self.headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
        }



    
    def scrape_zacks_rank(self, tag):
        rank = None
        url = "https://www.zacks.com/stock/quote/"+tag
        req = requests.get(url, self.headers)
        # web scraping with selenium
        # driver = webdriver.Firefox()
        # driver.get(url)
        
        soup = BeautifulSoup(driver, 'html.parser')
        rank = soup.find_all('p', class_="rank_view").get_text()
        print(str(rank))
        return rank


hs = HtmlScarpe()
hs.scrape_zacks_rank("FDX")
    
