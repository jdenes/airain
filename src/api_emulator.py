import time
import logging
from selenium import webdriver
from datetime import datetime, timedelta
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

logger = logging.getLogger(__name__)


class Emulator:

    def __init__(self, user_name, password,  mode='invest'):
        self.user_name = user_name
        self.password = password
        self.mode = mode
        self.driver = webdriver.Firefox(log_path='../logs/geckodriver.log')
        self.driver.get("https://demo.trading212.com/")
        time.sleep(6)
        self.driver.find_element_by_xpath("//input[@name='email']").send_keys(self.user_name)
        self.driver.find_element_by_xpath("//input[@name='password']").send_keys(self.password)
        self.driver.find_element_by_xpath("//input[@class='submit-button_input__3s_QD']").click()
        time.sleep(10)

    def open_trade(self, order):

        if self.mode == 'invest':
            if order['is_buy'] is None or not order['is_buy']:
                time.sleep(0.5)
                return self
            asset = f"{order['asset']}_US_EQ"
            xpath = f"//div[@data-rbd-draggable-id='{asset}']//span[@data-qa-trading-btn='btn-buy']"
            self.driver.find_element_by_xpath(xpath).click()
            self.driver.find_element_by_xpath("//input[@class='input css-jjd680']").click()
            ActionChains(self.driver).send_keys(order['quantity']).perform()
            time.sleep(0.5)
            xpath = f"//div[@data-qa-ticker='{asset}']//div[@class='button accent-button']"
            self.driver.find_element_by_xpath(xpath).click()
            xpath = "//div[@class='review-order']//div[@class='button accent-button']"
            self.driver.find_element_by_xpath(xpath).click()

        else:
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

        if self.mode == 'invest':

            self.driver.find_element_by_xpath("//div[@data-qa-table='pending-orders']").click()
            self.driver.find_element_by_xpath("//div[@data-qa-table='positions']").click()
            time.sleep(1)
            positions = self.driver.find_elements_by_xpath("//div[@class='positions-table-item']")
            tickers = [a.get_attribute("data-qa-ticker") for a in positions]

            if len(tickers) == 0:
                logger.warning("API emulator found no position to close")
                return self

            for ticker in tickers:
                xpath = f"//div[(@class='positions-table-item') and (@data-qa-ticker='{ticker}')]"
                self.driver.find_element_by_xpath(xpath).click()
                time.sleep(1)
                xpath = "//div[@class='investment web']//div[@class='label']"
                quantity = self.driver.find_element_by_xpath(xpath).text.split()[0]
                xpath = "//div[@class='invest-instrument-advanced-header']//span[@data-qa-trading-btn='btn-sell']"
                self.driver.find_element_by_xpath(xpath).click()
                self.driver.find_element_by_xpath("//input[@class='input css-jjd680']").click()
                ActionChains(self.driver).send_keys(quantity).perform()
                self.driver.find_element_by_xpath("//div[@class='button accent-button']").click()
                self.driver.find_element_by_xpath("//div[@class='button accent-button']").click()
                time.sleep(1)
                xpath = "//div[@class='svg-icon-holder close-button rectangular close-button-in-header']"
                self.driver.find_element_by_xpath(xpath).click()

        else:
            try:
                self.driver.find_element_by_xpath("//span[@class='account-panel-close-all svg-icon-holder']").click()
                self.driver.find_element_by_xpath("//div[@class='close-all-positions-button button blue-button']").click()
            except NoSuchElementException:
                logger.warning("API emulator found no position to close")

        time.sleep(3)
        return self

    def get_current_prices(self):

        if self.mode == 'invest':
            res = {'date': datetime.today().strftime('%Y-%m-%d %H:%M:%S')}
            xpath = "//div[@class='invest-tradebox _focusable pretty-name-shown']"
            open_prices = self.driver.find_elements_by_xpath(xpath)
            for elt in open_prices:
                asset = elt.find_element_by_xpath(".//span[@class='instrument-code']").text
                int_price = elt.find_element_by_xpath(".//span[@class='integer-value']").text
                dec_price = elt.find_element_by_xpath(".//span[@class='decimal-value']").text
                price = int(int_price) + float(dec_price)
                res[asset] = price
            return res

        else:
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
