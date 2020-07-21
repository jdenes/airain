import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


class Emulator:

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password
        self.driver = webdriver.Firefox(log_path='./logs/geckodriver.log')
        self.driver.get("https://demo.trading212.com/")
        self.driver.find_element_by_xpath("//input[@id='username-real']").send_keys(self.user_name)
        self.driver.find_element_by_xpath("//input[@id='pass-real']").send_keys(self.password)
        self.driver.find_element_by_xpath("//input[@class='button-login']").click()
        time.sleep(6)

    def open_trade(self, order):
        if order['is_buy'] is None:
            time.sleep(1)
            return self
        dir = 'buy' if order['is_buy'] is True else 'sell'
        xpath = f"//div[@data-code='{order['asset']}']//span[@class='buy-sell-price-container {dir}']"
        self.driver.find_element_by_xpath(xpath).click()
        self.driver.find_element_by_xpath("//div[@class='dropdown-arrow svg-icon-holder']").click()
        ActionChains(self.driver).send_keys(order['quantity']).perform()
        self.driver.find_element_by_xpath("//div[@class='custom-button confirm-button']").click()
        time.sleep(1)
        return self

    def close_all_trades(self):
        self.driver.find_element_by_xpath("//span[@class='account-panel-close-all svg-icon-holder']").click()
        self.driver.find_element_by_xpath("//div[@class='close-all-positions-button button blue-button']").click()
        time.sleep(1)
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

    def quit(self):
        self.driver.quit()
