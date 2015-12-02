import time
import sys
import requests
import os
from calendar import monthrange
from selenium import webdriver
from selenium.webdriver.support.ui import Select

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#if len(sys.argv) < 3:
##    print("usage: download.py YEAR COMPANY_CODES")
#    print("usage: download.py 2012 IBM MSFT DELL")
#    #sys.exit()

years = [2012]
tickers = ['xle']
months = [3]


# log in
driver = webdriver.Firefox()
driver.get('https://wrds-web.wharton.upenn.edu/wrds/index.cfm')
username_box = driver.find_element_by_id('username')
password_box = driver.find_element_by_id('password')
username_box.send_keys('xh2013')
password_box.send_keys('Xh66916202')
password_box.submit()


def download_quotes(ticker, year, month, day):
    driver.get('https://wrds-web.wharton.upenn.edu/wrds/ds/taq/cqm/index.cfm')
    company_codes = driver.find_element_by_id('code-lookup')
    company_codes.send_keys(ticker)

    # variables
    driver.find_element_by_id('var_BID').click()
    driver.find_element_by_id('var_BIDSIZ').click()
    driver.find_element_by_id('var_ASK').click()
    driver.find_element_by_id('var_ASKSIZ').click()
    driver.find_element_by_id('var_NATBBO_IND').click()
    driver.find_element_by_id('csv').click()
    driver.find_element_by_id('none').click()

    # time
    beg_m = Select(driver.find_element_by_id('beg_m'))
    beg_d = Select(driver.find_element_by_id('beg_d'))
    beg_yr = Select(driver.find_element_by_id('beg_yr'))
    end_m = Select(driver.find_element_by_id('end_m'))
    end_d = Select(driver.find_element_by_id('end_d'))
    end_yr = Select(driver.find_element_by_id('end_yr'))
    beg_m.select_by_index(month-1)
    end_m.select_by_index(month-1)
    beg_d.select_by_index(day-1)
    end_d.select_by_index(day-1)
    beg_yr.select_by_visible_text(str(year))
    end_yr.select_by_visible_text(str(year))
    end_hh = Select(driver.find_element_by_id('end_hh'))
    end_hh.select_by_visible_text('24')
    driver.find_element_by_id('form_submit').click()

    # switch focus to newly opened tab
    time.sleep(5)
    new_tab = driver.window_handles[-1]
    driver.close()
    driver.switch_to.window(new_tab)

    # wait for query to process
    start_time = time.time()
    result = None

    while ('No matches found.' not in driver.page_source
           and int(time.time() - start_time) < 240
           and result is None):

        time.sleep(5)
        try:
            result = driver.find_element_by_partial_link_text('.csv')
        except:
            pass

    if result is None:
        return

    all_cookies = driver.get_cookies()
    cookies = {}
    for s_cookie in all_cookies:
        cookies[s_cookie["name"]] = s_cookie["value"]

    url = result.get_attribute('href')
    response = requests.get(url, cookies=cookies, stream=True, verify=False)

    if not response.ok:
        print("Bad Response")
        return

    # download results
    filename = ticker.replace(" ", "_") + '_' + str(month).zfill(2) + '_' + str(day).zfill(2) + '_' + str(year)[2:]
    filename = filename + os.path.sep + filename + '_quotes.csv'
    filename = os.path.join(root_dir, 'data', filename)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as handle:
        for block in response.iter_content(1024):
            handle.write(block)


def download_trades(ticker, year, month, day):
    driver.get('https://wrds-web.wharton.upenn.edu/wrds/ds/taq/ctm/index.cfm')
    company_codes = driver.find_element_by_id('code-lookup')
    company_codes.send_keys(ticker)

    # variables
    driver.find_element_by_id('var_SIZE').click()
    driver.find_element_by_id('var_PRICE').click()
    driver.find_element_by_id('csv').click()
    driver.find_element_by_id('none').click()

    # time
    beg_m = Select(driver.find_element_by_id('beg_m'))
    beg_d = Select(driver.find_element_by_id('beg_d'))
    beg_yr = Select(driver.find_element_by_id('beg_yr'))
    end_m = Select(driver.find_element_by_id('end_m'))
    end_d = Select(driver.find_element_by_id('end_d'))
    end_yr = Select(driver.find_element_by_id('end_yr'))
    beg_m.select_by_index(month-1)
    end_m.select_by_index(month-1)
    beg_d.select_by_index(day-1)
    end_d.select_by_index(day-1)
    beg_yr.select_by_visible_text(str(year))
    end_yr.select_by_visible_text(str(year))
    end_hh = Select(driver.find_element_by_id('end_hh'))
    end_hh.select_by_visible_text('24')
    driver.find_element_by_id('form_submit').click()

    # switch focus to newly opened tab
    time.sleep(5)
    new_tab = driver.window_handles[-1]
    driver.close()
    driver.switch_to.window(new_tab)

    # wait for query to process
    start_time = time.time()
    result = None

    while ('No matches found.' not in driver.page_source
           and int(time.time() - start_time) < 240
           and result is None):

        time.sleep(5)
        try:
            result = driver.find_element_by_partial_link_text('.csv')
        except:
            pass

    if result is None:
        return

    all_cookies = driver.get_cookies()
    cookies = {}
    for s_cookie in all_cookies:
        cookies[s_cookie["name"]] = s_cookie["value"]

    url = result.get_attribute('href')
    response = requests.get(url, cookies=cookies, stream=True, verify=False)

    if not response.ok:
        print("Bad Response")
        return

    # download results
    filename = ticker.replace(" ", "_") + '_' + str(month).zfill(2) + '_' + str(day).zfill(2) + '_' + str(year)[2:]
    filename = filename + os.path.sep + filename + '_trades.csv'
    filename = os.path.join(root_dir, 'data', filename)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'wb') as handle:
        for block in response.iter_content(1024):
            handle.write(block)


# query
for year in years:
    for month in months:
        for day in range(monthrange(year, month)[1]):
            print("downloading " + str(month) + '/' + str(day+1) + '/'+str(year))
            for ticker in tickers:
                download_quotes(ticker, year, month, day+1)
                download_trades(ticker, year, month, day+1)

time.sleep(2)
driver.get('https://wrds-web.wharton.upenn.edu/wrds/?logout=true')
driver.quit()
