from selenium import webdriver


class Emulator:

    def __init__(self, user_name, password):
        self.user_name = user_name
        self.password = password
        self.driver = webdriver.Firefox()
        self.driver.get("https://demo.trading212.com/")
        self.driver.find_element_by_xpath("//input[@id='username-real']").send_keys(self.user_name)
        self.driver.find_element_by_xpath("//input[@id='pass-real']").send_keys(self.password)
        self.driver.find_element_by_xpath("//input[@class='button-login']").click()

    def open_trade(self):
        pass

    def close_all_trades(self):
        pass
