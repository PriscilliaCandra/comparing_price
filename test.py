from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.common.by import By
import time

options = Options()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()), options=options)

# URL Tokopedia (contoh)
url = 'https://www.tokopedia.com/search?st=product&q=pisang'
driver.get(url)

# Tunggu konten dimuat (bisa pakai WebDriverWait juga untuk lebih robust)
time.sleep(5)

# Ambil semua elemen produk
products = driver.find_elements(By.CLASS_NAME, 'css-5wh65g')

for product in products[:10]:
    try:
        # Nama produk
        name = product.find_element(By.XPATH, ".//span[contains(text(), '')]").text
    except:
        name = 'Nama tidak ditemukan'

    try:
        # Harga
        price = product.find_element(By.XPATH, ".//div[contains(text(), 'Rp')]").text
    except:
        price = 'Harga tidak ditemukan'

    try:
        # Jumlah terjual
        sold = product.find_element(By.XPATH, ".//span[contains(text(), 'terjual')]").text
    except:
        sold = 'Jumlah terjual tidak ditemukan'

    print({'Nama Produk': name, 'Harga': price, 'Terjual': sold})


driver.quit()
