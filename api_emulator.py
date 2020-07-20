import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


class Emulator:

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password
        self.driver = webdriver.Firefox()
        self.driver.get("https://demo.trading212.com/")
        self.driver.find_element_by_xpath("//input[@id='username-real']").send_keys(self.user_name)
        self.driver.find_element_by_xpath("//input[@id='pass-real']").send_keys(self.password)
        self.driver.find_element_by_xpath("//input[@class='button-login']").click()
        time.sleep(6)

    def open_trade(self, order):
        # order = {'asset': asset, 'is_buy': None, 'open': open, 'quantity': quantity, 'date': date}
        if order['is_buy'] is None:
            time.sleep(1)
            return -1
        dir = 'buy' if order['is_buy'] is True else 'sell'
        xpath = f"//div[@data-code='{order['asset']}']//span[@class='buy-sell-price-container {dir}']"
        self.driver.find_element_by_xpath(xpath).click()
        self.driver.find_element_by_xpath("//div[@class='dropdown-arrow svg-icon-holder']").click()
        ActionChains(self.driver).send_keys(order['quantity']).perform()
        self.driver.find_element_by_xpath("//div[@class='custom-button confirm-button']").click()
        time.sleep(1)
        return 0

    def close_all_trades(self):
        self.driver.find_element_by_xpath("//span[@class='account-panel-close-all svg-icon-holder']").click()
        self.driver.find_element_by_xpath("//div[@class='close-all-positions-button button blue-button']").click()
        time.sleep(1)
        return 1

    def quit(self):
        self.driver.quit()