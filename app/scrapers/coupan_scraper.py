# app/scrapers/coupan_scraper.py
import logging
import asyncio
import requests
import random # 추가
import time # 추가
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options # <<< 추가된 부분
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import UnexpectedAlertPresentException, TimeoutException, NoAlertPresentException # 예외 클래스 import
from typing import List, Dict, Any

# 스크래핑 시 주의사항:
# 1. 쿠팡의 robots.txt 파일을 확인하고 준수해야 합니다.
# 2. 웹사이트의 HTML 구조는 예고 없이 변경될 수 있으며, 이 경우 스크래퍼가 오작동할 수 있습니다.
# 3. 과도한 요청은 대상 웹사이트에 부담을 줄 수 있으므로, 적절한 요청 간격을 유지해야 합니다.
# 4. User-Agent를 설정하여 일반 브라우저처럼 보이도록 하는 것이 좋습니다.
# 5. 쿠팡은 동적 콘텐츠 로딩을 사용할 수 있으므로, requests/BeautifulSoup만으로는 한계가 있을 수 있습니다.
#    이 경우 Selenium, Playwright 등의 브라우저 자동화 도구 사용을 고려해야 할 수 있으나,
#    현재 프로젝트 구성 및 요청의 복잡도를 고려하여 우선 requests/BeautifulSoup으로 시도합니다.

# 예시: 선택자는 실제 쿠팡 사이트를 분석하여 정확하게 수정해야 합니다.
PRODUCT_NAME_SELECTOR = "h2.prod-buy-header__title"  # 상품명 CSS 선택자 (HTML에서 유효함 확인)
PRICE_SELECTOR = ".prod-buy-price .total-price strong"  # 가격 CSS 선택자 (할인된 최종 가격의 숫자 부분)
IMAGE_SELECTOR = "img#repImage"  # 대표 상품 이미지 CSS 선택자 (ID 사용)
REVIEW_SELECTOR = ""  # TODO: 실제 리뷰 컨테이너 또는 리뷰 수 선택자로 변경 필요
QNA_SELECTOR = ""  # TODO: 실제 Q&A 컨테이너 또는 Q&A 수 선택자로 변경 필요
REVIEW_TAB_SELECTOR = "" # TODO: 실제 리뷰 탭 버튼 선택자로 변경 필요
QNA_TAB_SELECTOR = "" # TODO: 실제 Q&A 탭 버튼 선택자로 변경 필요

logger = logging.getLogger(__name__)

def setup_driver():
    """Selenium WebDriver를 설정하고 반환합니다."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # GUI 없이 실행
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.add_argument("accept-language=ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(45)  # 페이지 로드 타임아웃 증가
        logger.info("[INFO] ChromeDriver setup successfully using webdriver_manager.")
        return driver
    except Exception as e:
        logger.error(f"[ERROR] Failed to setup ChromeDriver with webdriver_manager: {e}")
        raise  # 드라이버 설정 실패 시 예외 발생시켜 상위로 전파

async def scrape_product_details_from_coupan(product_url: str) -> Dict[str, Any]:
    logger.info(f"Scraping product details from: {product_url}")
    driver = None
    product_details: Dict[str, Any] = {
        "product_name": "N/A",
        "price": "N/A",
        "image_url": "N/A",
        "reviews": [],
        "qna": [],
        "product_url": product_url,
    }

    try:
        driver = await asyncio.to_thread(setup_driver)

        async def handle_alert_if_present(driver_instance, context_msg="알림 확인"):
            try:
                # switch_to.alert는 즉시 실행되지만, .text나 .accept()는 블로킹 호출일 수 있음
                alert = await asyncio.to_thread(lambda: driver_instance.switch_to.alert)
                alert_text = await asyncio.to_thread(lambda: alert.text) # alert.text는 블로킹 가능
                logger.info(f"알림창 감지됨 ({context_msg}): '{alert_text}'. 수락합니다.")
                await asyncio.to_thread(alert.accept) # alert.accept()는 블로킹 가능
                await asyncio.to_thread(time.sleep, random.uniform(0.5, 1.5)) # 알림창 닫힌 후 안정화 시간
                return True
            except NoAlertPresentException:
                # logger.debug(f"알림창 없음 ({context_msg}).") # 로그가 너무 많아질 수 있음
                return False
            except Exception as e_alert:
                logger.error(f"알림창 처리 중 오류 발생 ({context_msg}): {e_alert}")
                return False
        await asyncio.to_thread(driver.get, product_url)

        # --- 초기 알림창 처리 ---
        logger.info(f"Checking for initial alert after loading URL: {product_url}")
        await handle_alert_if_present(driver, "initial page load")
        logger.info("Initial alert check finished.")
        # --- 초기 알림창 처리 끝 ---


        # 페이지 로드 및 안정화 대기
        # 알림창 처리 후, 그리고 주요 요소 추출 전에 페이지가 완전히 로드되고 안정화될 시간을 추가로 줍니다.
        stabilization_time = random.uniform(5, 8) # 대기 시간을 5~8초로 늘림
        logger.info(f"Waiting for page stabilization for {stabilization_time:.2f} seconds before extracting product name. URL: {product_url}")
        await asyncio.to_thread(time.sleep, stabilization_time)
        logger.info(f"Page stabilization wait finished. URL: {product_url}") 

        # 상품명 추출 (WebDriverWait 사용)
        try:
            logger.info(f"Attempting to find product name with selector: {PRODUCT_NAME_SELECTOR} for URL: {product_url}")
            product_name_element = await asyncio.to_thread(
                WebDriverWait(driver, 30).until,  # 대기 시간 30초로 증가
                EC.presence_of_element_located((By.CSS_SELECTOR, PRODUCT_NAME_SELECTOR))
            )
            product_details["product_name"] = product_name_element.text.strip()
            logger.info(f"Found product name: {product_details['product_name']} for URL: {product_url}")
        except TimeoutException:
            logger.warning(f"Product name not found (timed out after 30s) for {product_url} using selector '{PRODUCT_NAME_SELECTOR}'.")
            try:
                page_source_snippet = await asyncio.to_thread(lambda: driver.page_source[:5000]) # 페이지 소스 앞 5000자
                logger.info(f"Page source snippet at Timeout for product name (first 5000 chars):\n{page_source_snippet}")
            except Exception as ps_e:
                logger.error(f"Failed to get page source on TimeoutException for product name: {ps_e}")
            await handle_alert_if_present(driver, "product name scraping (TimeoutException)")
        except UnexpectedAlertPresentException as unexp_alert_e:
            alert_text_from_exception = unexp_alert_e.alert_text if hasattr(unexp_alert_e, 'alert_text') and unexp_alert_e.alert_text else "N/A"
            logger.error(f"UnexpectedAlertPresentException while finding product name for {product_url}. Alert text from exception: {alert_text_from_exception}. Selector: '{PRODUCT_NAME_SELECTOR}'.")
            logger.debug(f"UnexpectedAlertPresentException details: {unexp_alert_e}")
            try:
                # 알림창이 이미 열려 있으므로 직접 처리 시도
                alert = driver.switch_to.alert
                actual_alert_text = alert.text
                logger.info(f"Accepting alert directly in UnexpectedAlertPresentException handler. Actual alert text: {actual_alert_text}")
                alert.accept()
                logger.info("Alert accepted. Logging page source afterwards.")
                # 알림창 처리 후 페이지 소스 로깅
                page_source_after_alert = await asyncio.to_thread(lambda: driver.page_source[:5000])
                logger.info(f"Page source snippet after accepting alert in UnexpectedAlertPresentException (first 5000 chars):\n{page_source_after_alert}")
            except NoAlertPresentException:
                logger.warning("NoAlertPresentException caught while trying to handle UnexpectedAlertPresentException - alert might have been dismissed already or by another handler.")
            except Exception as alert_handling_e:
                logger.error(f"Error handling alert within UnexpectedAlertPresentException block: {alert_handling_e}")
        except Exception as e:
            logger.error(f"Error finding product name for {product_url} using selector '{PRODUCT_NAME_SELECTOR}': {e}")
            try:
                page_source_snippet = await asyncio.to_thread(lambda: driver.page_source[:5000])
                logger.info(f"Page source snippet on general exception for product name (first 5000 chars):\n{page_source_snippet}")
            except Exception as ps_e:
                logger.error(f"Failed to get page source on general exception for product name: {ps_e}")
            await handle_alert_if_present(driver, "product name scraping (General Exception)")

        # 가격 추출
        if PRICE_SELECTOR:
            try:
                # 가격 가져오기
                logger.info(f"Attempting to find price with selector: {PRICE_SELECTOR} for URL: {product_url}")
                price_element = WebDriverWait(driver, 30).until( # 대기 시간 30초로 증가
                    EC.presence_of_element_located((By.CSS_SELECTOR, PRICE_SELECTOR)) # presence_of_element_located로 변경
                )
                logger.info(f"Price element found (present in DOM) for URL: {product_url}. Raw text from element: '{price_element.text}'")
                
                # Selenium의 .text 속성과 JavaScript의 textContent를 모두 시도
                price_text_selenium = price_element.text.strip() if price_element.text else ""
                
                # JavaScript를 사용하여 textContent 가져오기 (좀 더 안정적일 수 있음)
                price_text_js = ""
                try:
                    # driver.execute_script는 동기 함수이므로 asyncio.to_thread로 감싸야 합니다.
                    price_text_js = await asyncio.to_thread(driver.execute_script, "return arguments[0].textContent;", price_element)
                    price_text_js = price_text_js.strip() if price_text_js else ""
                except Exception as js_exc:
                    logger.warning(f"Failed to get price text using JavaScript for URL {product_url}: {js_exc}")

                logger.info(f"Price text (Selenium): '{price_text_selenium}', Price text (JS): '{price_text_js}' for URL: {product_url}")

                # Selenium .text 우선, 없거나 비정상적(숫자 없음)이면 JS 사용
                price_text_to_parse = price_text_selenium
                # Selenium 결과가 비어있거나, 청소 후 숫자가 아니면 JS 결과 시도
                if not price_text_selenium or not price_text_selenium.replace(",", "").replace("원", "").isdigit():
                    if price_text_js and price_text_js.replace(",", "").replace("원", "").isdigit(): # JS 결과가 있고, 청소 후 숫자이면 사용
                         logger.info(f"Selenium text for price was empty or non-numeric ('{price_text_selenium}'), using JS text: '{price_text_js}' for URL: {product_url}")
                         price_text_to_parse = price_text_js
                    else: # JS 결과도 부적절하면, 원래 Selenium 결과 사용 (로깅용)
                         logger.info(f"Both Selenium ('{price_text_selenium}') and JS ('{price_text_js}') text for price are problematic or non-numeric. Sticking with Selenium's original for parsing attempt for URL: {product_url}")
                         # price_text_to_parse는 이미 price_text_selenium으로 설정되어 있음
                
                if not price_text_to_parse: # 최종적으로 파싱할 텍스트가 없으면
                     logger.warning(f"Price text is empty for selector: {PRICE_SELECTOR}. URL: {product_url}")
                     # product_details["price"]는 "N/A"로 유지됨
                else:
                    cleaned_price_text = price_text_to_parse.replace(",", "").replace("원", "")
                    if cleaned_price_text.isdigit():
                        product_details["price"] = int(cleaned_price_text)
                        logger.info(f"Price parsed successfully: {product_details['price']} from '{price_text_to_parse}' for URL: {product_url}")
                    else:
                        # 숫자가 아닌 경우, 원본 텍스트를 저장하고 경고 로깅
                        logger.warning(f"Price text found ('{price_text_to_parse}') but could not be parsed as integer after cleaning ('{cleaned_price_text}'). Selector: {PRICE_SELECTOR}. URL: {product_url}")
                        product_details["price"] = price_text_to_parse # 파싱 실패시 원본 텍스트 저장
            except TimeoutException:
                logger.warning(f"Price not found (timed out after 30s) using selector: {PRICE_SELECTOR} with presence_of_element_located. URL: {product_url}")
                try:
                    page_source_snippet = await asyncio.to_thread(lambda: driver.page_source[:5000]) # 페이지 소스 앞 5000자
                    logger.info(f"Page source snippet at Timeout for price (first 5000 chars):\n{page_source_snippet}")
                except Exception as ps_e:
                    logger.error(f"Failed to get page source on TimeoutException for price: {ps_e}")
                await handle_alert_if_present(driver, "price scraping (TimeoutException)") # 컨텍스트 메시지 추가
            except Exception as e:
                logger.error(f"Error finding or parsing price with selector {PRICE_SELECTOR}: {e}. URL: {product_url}")
                await handle_alert_if_present(driver) # 알림창 확인 추가

        # 이미지 URL 추출
        if IMAGE_SELECTOR:
            try:
                image_element = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, IMAGE_SELECTOR))
                )
                product_details["image_url"] = image_element.get_attribute('src')
                if product_details["image_url"] and product_details["image_url"].startswith("//"):
                    product_details["image_url"] = "https" + product_details["image_url"]
                logger.info(f"Found image URL: {product_details['image_url']}")
            except TimeoutException:
                logger.warning(f"Image URL not found for {product_url} using selector '{IMAGE_SELECTOR}': TimeoutException")
            except Exception as e:
                logger.warning(f"Image URL not found for {product_url} using selector '{IMAGE_SELECTOR}': {e}")

        # 리뷰 추출 (WebDriverWait 및 탭 클릭 예시 - 실제 사이트 구조에 맞게 수정 필요)
        if REVIEW_TAB_SELECTOR and REVIEW_SELECTOR:
            try:
                review_elements_present = await asyncio.to_thread(
                    WebDriverWait(driver, 15).until,
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, REVIEW_SELECTOR))
                )

                page_source_after_interaction = await asyncio.to_thread(getattr, driver, 'page_source')
                soup_reviews = BeautifulSoup(page_source_after_interaction, "html.parser")

                reviews = soup_reviews.select(REVIEW_SELECTOR)
                if reviews:
                    for review in reviews:
                        product_details["reviews"].append({"text": review.text.strip()})  # 임시로 전체 텍스트
                    logger.info(f"Found {len(product_details['reviews'])} reviews for {product_url}")
                else:
                    logger.warning(f"No reviews found for {product_url} with selector '{REVIEW_SELECTOR}'. They might be dynamically loaded or selectors need update.")
            except Exception as e:
                logger.error(f"Error scraping reviews for {product_url}: {e}")
                logger.warning(f"No reviews found for {product_url}. They might be dynamically loaded or selectors need update.")
        else:
            logger.warning(f"Review selectors not fully defined for {product_url}. Skipping review scraping.")
            product_details["reviews"] = [] # 또는 [{'error': 'Review selectors not defined'}]

        # Q&A 추출 (WebDriverWait 및 탭 클릭 예시 - 실제 사이트 구조에 맞게 수정 필요)
        if QNA_TAB_SELECTOR and QNA_SELECTOR:
            try:
                qna_elements_present = await asyncio.to_thread(
                    WebDriverWait(driver, 15).until,
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, QNA_SELECTOR))
                )

                page_source_after_qna_interaction = await asyncio.to_thread(getattr, driver, 'page_source')
                soup_qna = BeautifulSoup(page_source_after_qna_interaction, "html.parser")

                qna_items = soup_qna.select(QNA_SELECTOR)
                if qna_items:
                    for item in qna_items:
                        product_details["qna"].append({"text": item.text.strip()})  # 임시로 전체 텍스트
                    logger.info(f"Found {len(product_details['qna'])} Q&A items for {product_url}")
                else:
                    logger.warning(f"No Q&A found for {product_url} with selector '{QNA_SELECTOR}'. They might be dynamically loaded or selectors need update.")
            except Exception as e:
                logger.error(f"Error scraping Q&A for {product_url}: {e}")
                logger.warning(f"No Q&A found for {product_url}. They might be dynamically loaded or selectors need update.")
        else:
            logger.warning(f"Q&A selectors not fully defined for {product_url}. Skipping Q&A scraping.")
            product_details["qna"] = [] # 또는 [{'error': 'Q&A selectors not defined'}]

    except UnexpectedAlertPresentException as e:
        logger.error(f"Critical UnexpectedAlertPresentException during scraping {product_url}: {e.alert_text if e.alert_text else str(e)}")
        product_details["error"] = f"UnexpectedAlertPresentException: {e.alert_text if e.alert_text else str(e)}"
        if driver:
            try:
                # 예외 발생 시 알림창을 닫으려고 시도
                alert = driver.switch_to.alert
                alert.accept()
            except Exception as alert_e:
                logger.error(f"Could not dismiss alert during critical error handling: {alert_e}")
    except Exception as e:
        logger.error(f"An error occurred while scraping {product_url}: {e}")
        product_details["error"] = str(e)
    finally:
        if driver:
            await asyncio.to_thread(driver.quit)
            logger.info(f"WebDriver closed for product details page: {product_url}")

    if product_details["product_name"] == "N/A" and product_details["price"] == "N/A" and product_details["image_url"] == "N/A":
        logger.warning(f"Failed to scrape significant details for {product_url}")
        # product_details["error"]가 이미 설정되었을 수 있으므로, 덮어쓰지 않도록 주의
        if "error" not in product_details:
             product_details["error"] = "Failed to scrape significant details (name, price, image)."

    return product_details

async def get_coupan_report_for_bono_house() -> List[Dict[str, Any]]:
    """
    "보노 하우스" 키워드로 쿠팡에서 상품 정보를 스크랩하여 보고서 형태로 반환합니다.
    """
    keyword = "보노 하우스"
    report_data = []
    max_products_to_scrape = 3 # 실제 스크랩할 최대 상품 수 제한

    print(f"[INFO] Starting Coupang report generation for keyword: '{keyword}'")
    product_page_urls = await search_products_on_coupan(keyword, max_pages=1) # 검색은 우선 1페이지만

    if not product_page_urls:
        print(f"[WARN] No products found for keyword: '{keyword}'. Returning mock data as fallback.")
        # 목업 데이터 반환 로직은 아래로 이동
    else:
        print(f"[INFO] Will attempt to scrape details for up to {max_products_to_scrape} products.")
        for i, product_url in enumerate(product_page_urls[:max_products_to_scrape]):
            print(f"[INFO] Scraping product {i+1}/{len(product_page_urls[:max_products_to_scrape])}: {product_url}")
            details = await scrape_product_details_from_coupan(product_url)
            if details.get("product_name"): # 유효한 상품명이 있는 경우에만 추가
                report_data.append(details)
            
            # 마지막 요청이 아니면 대기
            if i < len(product_page_urls[:max_products_to_scrape]) - 1:
                await asyncio.sleep(random.uniform(2.0, 4.0)) # 요청 간 충분한 대기

    # 스크랩된 데이터가 없거나, 유의미한 데이터가 없는 경우 목업 데이터 사용
    if not report_data or not any(item.get("product_name") for item in report_data):
        print(f"[WARN] No actual data scraped or data was insufficient for '{keyword}'. Using MOCK DATA as fallback.")
        report_data = [
            {
                "product_name": "[보노하우스] 따뜻한 극세사 겨울 침구세트 (Q) (Mock)",
                "product_url": "https://www.coupang.com/vp/products/12345_mock",
                "reviews": [
                    {"author": "김*정", "rating": 5, "content": "너무 따뜻하고 좋아요! 색상도 예쁩니다.", "date": "2024.12.15"},
                    {"author": "이*라", "rating": 4, "content": "가격 대비 만족합니다. 배송도 빨랐어요.", "date": "2024.12.10"}
                ],
                "qna": [
                    {"question": "세탁은 어떻게 하나요?", "answer": "찬물에 중성세제로 단독세탁 권장드립니다."},
                    {"question": "다른 색상도 있나요?", "answer": "네, 상세페이지 하단에서 확인 가능합니다."}
                ]
            },
            {
                "product_name": "보노하우스 어린이 소프트 블록 100pcs (Mock)",
                "product_url": "https://www.coupang.com/vp/products/67890_mock",
                "reviews": [
                    {"author": "박*수", "rating": 5, "content": "아이가 정말 좋아해요. 안전하고 퀄리티도 좋습니다.", "date": "2025.01.02"}
                ],
                "qna": []
            }
        ]
        print(f"[INFO] Generated mock Coupang report for '{keyword}' with {len(report_data)} items.")
    else:
        print(f"[INFO] Successfully generated Coupang report for '{keyword}' with {len(report_data)} actual items.")
    
    return report_data

# === 테스트 코드 시작 (이전 테스트 코드가 있다면 이 블록으로 교체해주세요) ===
if __name__ == "__main__":
    import asyncio
    # 로깅 기본 설정 (콘솔에서 로그를 보기 위함)
    # 파일 상단에 이미 logger = logging.getLogger(__name__) 가 정의되어 있어야 합니다.
    # import logging 구문도 파일 상단에 있어야 합니다.
    if not logging.getLogger().hasHandlers(): # 핸들러가 이미 설정되어 있는지 확인
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    async def main_test_single_item(): # 함수 이름을 변경하여 혼동 방지
        # 테스트하고 싶은 쿠팡 상품의 전체 URL을 여기에 입력하세요.
        # 예시 URL (이전에 제공해주신 "보노하우스" 상품 URL):
        test_url = "https://www.coupang.com/vp/products/6956415718?itemId=17014918361&vendorItemId=84190638820&sourceType=srp_product_ads&clickEventId=864d6d00-375e-11f0-9563-917404c266cc&korePlacement=15&koreSubPlacement=1&q=%EB%B3%B4%EB%85%B8%ED%95%98%EC%9A%B0%EC%8A%A4&itemsCount=36&searchId=77307e8a548098&rank=0&searchRank=0&isAddedCart="
        # test_url = "다른_쿠팡_상품_URL_입력" # 다른 상품으로 테스트 시 이 부분을 수정하세요.
        
        logger.info(f"단일 상품 스크래핑 테스트 시작. URL: {test_url}")
        details = await scrape_product_details_from_coupan(test_url)
        
        print("\n--- 스크랩된 상품 상세 정보 ---")
        if details:
            print(f"  상품명: {details.get('product_name', 'N/A')}")
            print(f"  가격: {details.get('price', 'N/A')}")
            print(f"  이미지 URL: {details.get('image_url', 'N/A')}")
            
            reviews_data = details.get('reviews', ["N/A"])
            print(f"  리뷰 (최대 5개):")
            if reviews_data and reviews_data != ["N/A"] and reviews_data != []:
                for review in reviews_data:
                    review_text = str(review)
                    print(f"    - {review_text[:100]}{'...' if len(review_text) > 100 else ''}")
            else:
                print(f"    - 리뷰 정보 없음 또는 가져오지 못함 (REVIEW_SELECTOR: '{REVIEW_SELECTOR}')")

            qna_data = details.get('qna', ["N/A"])
            print(f"  Q&A (최대 5개):")
            if qna_data and qna_data != ["N/A"] and qna_data != []:
                for q_item in qna_data:
                    q_text = str(q_item)
                    print(f"    - {q_text[:100]}{'...' if len(q_text) > 100 else ''}")
            else:
                print(f"    - Q&A 정보 없음 또는 가져오지 못함 (QNA_SELECTOR: '{QNA_SELECTOR}')")
            
            print(f"  상품 URL: {details.get('product_url', 'N/A')}")
            if details.get('error'):
                print(f"  오류: {details.get('error')}")
        else:
            print("  상세 정보를 가져오지 못했습니다.")
        print("-----------------------------")

    asyncio.run(main_test_single_item())
# === 테스트 코드 끝 ===
