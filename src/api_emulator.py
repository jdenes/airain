import time
import logging
from selenium import webdriver
from datetime import datetime, timedelta
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

logger = logging.getLogger(__name__)


class Emulator:

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password
        self.driver = webdriver.Firefox(log_path='../logs/geckodriver.log')
        self.driver.get("https://demo.trading212.com/")
        self.driver.find_element_by_xpath("//input[@id='username-real']").send_keys(self.user_name)
        self.driver.find_element_by_xpath("//input[@id='pass-real']").send_keys(self.password)
        self.driver.find_element_by_xpath("//input[@class='button-login']").click()
        time.sleep(20)

    def open_trade(self, order):
        if order['is_buy'] is None:
            time.sleep(1)
            return self
        direction = 'buy' if order['is_buy'] == 1 else 'sell'
        xpath = f"//div[@data-code='{order['asset']}']//span[@class='buy-sell-price-container {direction}']"
        self.driver.find_element_by_xpath(xpath).click()
        self.driver.find_element_by_xpath("//div[@class='dropdown-arrow svg-icon-holder']").click()
        ActionChains(self.driver).send_keys(order['quantity']).perform()
        self.driver.find_element_by_xpath("//div[@class='custom-button confirm-button']").click()
        time.sleep(1)
        return self

    def close_all_trades(self):
        try:
            self.driver.find_element_by_xpath("//span[@class='account-panel-close-all svg-icon-holder']").click()
            self.driver.find_element_by_xpath("//div[@class='close-all-positions-button button blue-button']").click()
        except NoSuchElementException:
            logger.info("API emulator found no position to close")
        time.sleep(3)
        return self

    def get_open_prices(self):
        res = {'date': datetime.today().strftime('%Y-%m-%d')}
        xpath = "//tbody[@class='table-body dataTable-show-currentprice-arrows']//tr"
        open_prices = self.driver.find_elements_by_xpath(xpath)
        for elt in open_prices:
            asset = elt.get_attribute("data-code")
            price = elt.find_element_by_xpath("./td[@class='averagePrice']").text
            res[asset] = price
        return res

    def get_trades_results(self, horizon=3):
        res = []
        time_format = '%d.%m.%Y %H:%M:%S'
        self.driver.find_element_by_xpath("//span[@class='button-arrow svg-icon-holder']").click()
        self.driver.find_element_by_xpath("//div[text()='Reports']").click()
        self.driver.find_element_by_xpath("//div[@class='item item-account-menu-reports-result']").click()
        time.sleep(5)
        dates_list = [datetime.today()]
        while min(dates_list) >= datetime.today() - timedelta(days=horizon):
            rows = self.driver.find_elements_by_xpath("//tbody[@class='table-body']/tr")
            dates_list = [datetime.strptime(r.find_elements_by_xpath("./td")[8].text, time_format) for r in rows]
            for row in rows:
                cells = row.find_elements_by_xpath("./td")
                asset = row.find_element_by_xpath("./td[@class='clickable-id']").get_attribute("data-code")
                quantity = int(cells[2].text.replace(' ', ''))
                is_buy = 1 if cells[3].text == 'Buy' else 0
                result = float(cells[6].text)
                open_price = float(cells[4].text)
                close_price = float(cells[5].text)
                date = datetime.strptime(cells[8].text, time_format).strftime('%Y-%m-%d')
                res.append({'date': date, 'asset': asset, 'is_buy': is_buy, 'quantity': quantity,
                            'result': result, 'open': open_price, 'close': close_price})
            try:
                self.driver.find_element_by_xpath("//span[@class='button' and "
                                                  "@data-dojo-attach-point='nextButton']").click()
            except NoSuchElementException:
                break
        return res

    def quit(self):
        self.driver.quit()
