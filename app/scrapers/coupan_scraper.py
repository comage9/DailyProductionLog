# app/scrapers/coupan_scraper.py

# TODO: Future Codebase Improvements (Scraper Specific)
# 1. Standardize Comments and Identifiers:
#    - Convert all comments, variable names, and function names to English for consistency and broader collaboration.
# 2. Enhance Selector Robustness:
#    - Explore using more stable selectors, such as XPath expressions based on text content or more specific attributes,
#      to reduce breakage when Coupang updates its HTML structure.
#    - Consider using a library or pattern that allows defining multiple fallback selectors for critical elements.
# 3. Improve Anti-Scraping Evasion:
#    - Implement proxy rotation (e.g., using a pool of proxies from a service or private list).
#    - Employ more sophisticated user-agent management (e.g., rotating through a list of realistic user agents).
#    - Consider adding delays or randomized behavior that more closely mimics human interaction if IP blocking or CAPTCHAs become persistent.
# 4. Modularize Scraping Logic:
#    - Break down `scrape_product_details_from_coupan` into smaller functions for each section (e.g., _scrape_product_info, _scrape_reviews, _scrape_qna).
# 5. Search Functionality:
#    - Implement the `search_products_on_coupan` function which is currently commented out in `get_coupan_report_for_bono_house`.
#      This would involve navigating Coupang's search results pages.

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

# CSS Selectors ( 쿠팡 웹사이트 구조 변경 시 업데이트 필요 )
# 주의: 이 선택자들은 예시이며, 실제 쿠팡 웹사이트와 다를 수 있습니다.
# 실제 사용 전 반드시 개발자 도구로 확인하고 업데이트해야 합니다.

# 상품 기본 정보
PRODUCT_NAME_SELECTOR = "h2.prod-buy-header__title"
PRICE_SELECTOR = ".prod-buy-price .total-price strong"
IMAGE_SELECTOR = "img#repImage"

# 탭 선택자
REVIEW_TAB_SELECTOR = "a[name='review']" # 예: <a name="review" href="...">리뷰</a>
QNA_TAB_SELECTOR = "a[name='productInquiry']"    # 예: <a name="productInquiry" href="...">문의</a>

# 리뷰 관련 선택자
REVIEW_CONTAINER_SELECTOR = "article.sdp-review__article__list" # 각 리뷰 아이템을 포함하는 전체 컨테이너
REVIEW_ITEM_SELECTOR = "article.sdp-review__article__list__review__item" # 개별 리뷰 아이템
REVIEW_AUTHOR_SELECTOR = "span.sdp-review__article__list__info__user__name"
REVIEW_RATING_SELECTOR = "div.sdp-review__article__list__info__product-info__star-gray > span" # width style로 별점 계산
REVIEW_DATE_SELECTOR = "div.sdp-review__article__list__info__product-info__reg-date"
REVIEW_CONTENT_SELECTOR = "div.sdp-review__article__list__review__content"
REVIEW_SHOW_MORE_BUTTON_SELECTOR = "button.sdp-review__article__page__more__button" # "더보기" 버튼
REVIEW_PAGINATION_SELECTOR = "button.sdp-review__article__page__num" # 페이지 번호 버튼들 (다음 페이지 클릭용)

# Q&A 관련 선택자
QNA_CONTAINER_SELECTOR = "div.product-qna__list-container" # Q&A 목록을 포함하는 전체 컨테이너
QNA_ITEM_SELECTOR = "article.product-qna__item" # 개별 Q&A 아이템
QNA_QUESTION_SELECTOR = "div.product-qna__item__question__text"
QNA_ANSWER_SELECTOR = "div.product-qna__item__answer__text"
QNA_AUTHOR_SELECTOR = "span.product-qna__item__writer-name" # 질문 작성자
QNA_DATE_SELECTOR = "span.product-qna__item__date" # 질문 날짜
QNA_SHOW_MORE_BUTTON_SELECTOR = "button.product-qna__page__more-button" # Q&A "더보기"
QNA_PAGINATION_SELECTOR = "button.product-qna__page__num" # Q&A 페이지 번호 버튼

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
        "scrape_status": { # Enhanced error logging field
            "details": "pending",
            "reviews": "pending",
            "qna": "pending"
        }
    }

    try:
        driver = await asyncio.to_thread(setup_driver)

        current_context = "initial page load"
        async def handle_alert_if_present(driver_instance, context_msg="알림 확인"):
            nonlocal current_context # Allow modification of outer scope variable
            current_context = context_msg # Update current context for error logging
            try:
                alert = await asyncio.to_thread(lambda: driver_instance.switch_to.alert)
                alert_text = await asyncio.to_thread(lambda: alert.text)
                logger.info(f"알림창 감지됨 ({current_context}): '{alert_text}'. 수락합니다.")
                await asyncio.to_thread(alert.accept)
                await asyncio.to_thread(time.sleep, random.uniform(0.5, 1.5))
                return True
            except NoAlertPresentException:
                return False
            except Exception as e_alert:
                logger.error(f"알림창 처리 중 오류 발생 ({current_context}): {e_alert}")
                return False
        
        await asyncio.to_thread(driver.get, product_url)
        await handle_alert_if_present(driver, "initial page load")
        
        stabilization_time = random.uniform(3, 5) # 기본 안정화 시간
        logger.info(f"Waiting for page stabilization for {stabilization_time:.2f} seconds. URL: {product_url}")
        await asyncio.to_thread(time.sleep, stabilization_time)
        
        # 상품 기본 정보 스크래핑
        current_context = "product name scraping"

        try:
            product_name_element = await asyncio.to_thread(
                WebDriverWait(driver, 20).until, # Increased timeout
                EC.presence_of_element_located((By.CSS_SELECTOR, PRODUCT_NAME_SELECTOR))
            )
            product_details["product_name"] = product_name_element.text.strip()
            product_details["scrape_status"]["details"] = "success_product_name"
        except TimeoutException:
            logger.warning(f"Timeout: Product name not found for {product_url} using selector '{PRODUCT_NAME_SELECTOR}'. Context: {current_context}")
            product_details["scrape_status"]["details"] = f"failure_product_name: TimeoutException on selector {PRODUCT_NAME_SELECTOR}"
            await handle_alert_if_present(driver, "product name scraping (TimeoutException)")
        except Exception as e:
            logger.error(f"Error finding product name for {product_url} using selector '{PRODUCT_NAME_SELECTOR}'. Context: {current_context}. Error: {e}")
            product_details["scrape_status"]["details"] = f"failure_product_name: {type(e).__name__} on selector {PRODUCT_NAME_SELECTOR}"
            await handle_alert_if_present(driver, "product name scraping (General Exception)")
        
        current_context = "price scraping"
        if PRICE_SELECTOR:
            try:
                price_element = await asyncio.to_thread(
                    WebDriverWait(driver, 20).until, # Increased timeout
                    EC.presence_of_element_located((By.CSS_SELECTOR, PRICE_SELECTOR))
                )
                price_text_selenium = price_element.text.strip() if price_element.text else ""
                price_text_js = await asyncio.to_thread(driver.execute_script, "return arguments[0].textContent;", price_element)
                price_text_js = price_text_js.strip() if price_text_js else ""
                
                price_text_to_parse = price_text_selenium
                if not price_text_selenium or not price_text_selenium.replace(",", "").replace("원", "").isdigit():
                    if price_text_js and price_text_js.replace(",", "").replace("원", "").isdigit():
                         price_text_to_parse = price_text_js
                
                if price_text_to_parse:
                    cleaned_price = price_text_to_parse.replace(",", "").replace("원", "")
                    if cleaned_price.isdigit():
                        product_details["price"] = int(cleaned_price)
                        product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + "; success_price"
                    else:
                        product_details["price"] = price_text_to_parse # Store original if not parsable
                        logger.warning(f"Price text '{price_text_to_parse}' not parsable for {product_url}. Context: {current_context}")
                        product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + f"; failure_price_parse: '{price_text_to_parse}'"
                else:
                    logger.warning(f"Price text empty for {product_url}. Context: {current_context}")
                    product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + "; failure_price_empty"
            except TimeoutException:
                logger.warning(f"Timeout: Price not found for {product_url} using selector '{PRICE_SELECTOR}'. Context: {current_context}")
                product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + f"; failure_price_timeout: {PRICE_SELECTOR}"
                await handle_alert_if_present(driver, "price scraping (TimeoutException)")
            except Exception as e:
                logger.error(f"Error finding price for {product_url} using selector '{PRICE_SELECTOR}'. Context: {current_context}. Error: {e}")
                product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + f"; failure_price_exception: {type(e).__name__} on {PRICE_SELECTOR}"
                await handle_alert_if_present(driver, "price scraping (General Exception)")

        current_context = "image URL scraping"
        if IMAGE_SELECTOR:
            try:
                image_element = await asyncio.to_thread(
                    WebDriverWait(driver, 10).until, # Shorter timeout for image
                    EC.presence_of_element_located((By.CSS_SELECTOR, IMAGE_SELECTOR))
                )
                product_details["image_url"] = image_element.get_attribute('src')
                if product_details["image_url"] and product_details["image_url"].startswith("//"):
                    product_details["image_url"] = "https" + product_details["image_url"]
                product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + "; success_image_url"
            except TimeoutException:
                logger.warning(f"Timeout: Image URL not found for {product_url} using selector '{IMAGE_SELECTOR}'. Context: {current_context}")
                product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + f"; failure_image_url_timeout: {IMAGE_SELECTOR}"
            except Exception as e:
                logger.warning(f"Error finding image URL for {product_url} using selector '{IMAGE_SELECTOR}'. Context: {current_context}. Error: {e}")
                product_details["scrape_status"]["details"] = product_details["scrape_status"].get("details", "") + f"; failure_image_url_exception: {type(e).__name__} on {IMAGE_SELECTOR}"
        
        # --- 리뷰 스크래핑 ---
        current_context = "reviews tab click"
        if REVIEW_TAB_SELECTOR and REVIEW_ITEM_SELECTOR:
            try:
                review_tab_button = await asyncio.to_thread(
                    WebDriverWait(driver, 15).until,
                    EC.element_to_be_clickable((By.CSS_SELECTOR, REVIEW_TAB_SELECTOR))
                )
                await asyncio.to_thread(driver.execute_script, "arguments[0].click();", review_tab_button)
                await asyncio.to_thread(time.sleep, random.uniform(2, 4)) # 탭 변경 후 로드 대기

                current_context = "reviews scraping"
                page_count = 0
                max_review_pages = 5 # 최대 리뷰 페이지 수 (무한 스크롤 방지)
                
                while page_count < max_review_pages:
                    await asyncio.to_thread(driver.execute_script, "window.scrollTo(0, document.body.scrollHeight);") # 페이지 끝까지 스크롤
                    await asyncio.to_thread(time.sleep, random.uniform(1, 2)) # 스크롤 후 로드 대기
                    
                    soup = BeautifulSoup(await asyncio.to_thread(lambda: driver.page_source), "html.parser")
                    review_elements = soup.select(REVIEW_ITEM_SELECTOR)
                    
                    current_reviews_count = len(product_details["reviews"])
                    for review_el in review_elements[current_reviews_count:]: # 새로 로드된 리뷰만 추가
                        author = review_el.select_one(REVIEW_AUTHOR_SELECTOR).text.strip() if review_el.select_one(REVIEW_AUTHOR_SELECTOR) else "N/A"
                        rating_style = review_el.select_one(REVIEW_RATING_SELECTOR)['style'] if review_el.select_one(REVIEW_RATING_SELECTOR) else "width:0%"
                        rating = int(float(rating_style.split("width:")[1].split("%")[0]) / 20) if "width:" in rating_style else 0 # 100% = 5 stars
                        date = review_el.select_one(REVIEW_DATE_SELECTOR).text.strip() if review_el.select_one(REVIEW_DATE_SELECTOR) else "N/A"
                        content = review_el.select_one(REVIEW_CONTENT_SELECTOR).text.strip() if review_el.select_one(REVIEW_CONTENT_SELECTOR) else "N/A"
                        product_details["reviews"].append({
                            "author": author, "rating": rating, "date": date, "content": content
                        })
                    
                    try: # "더보기" 또는 페이지네이션 버튼 클릭
                        show_more_button = await asyncio.to_thread(
                            WebDriverWait(driver, 5).until, # 더보기 버튼은 짧게 대기
                            EC.element_to_be_clickable((By.CSS_SELECTOR, REVIEW_SHOW_MORE_BUTTON_SELECTOR))
                        )
                        await asyncio.to_thread(driver.execute_script, "arguments[0].click();", show_more_button)
                        logger.info(f"Clicked review 'show more' button for {product_url}. Page {page_count + 1}")
                        await asyncio.to_thread(time.sleep, random.uniform(2, 3))
                    except TimeoutException: # 더보기 버튼이 없으면 페이지네이션 시도
                        try:
                            page_buttons = await asyncio.to_thread(lambda: driver.find_elements(By.CSS_SELECTOR, REVIEW_PAGINATION_SELECTOR))
                            next_page_button = next((btn for btn in page_buttons if btn.text == str(page_count + 2)), None) # 다음 페이지 번호
                            if next_page_button and next_page_button.is_enabled():
                                await asyncio.to_thread(driver.execute_script, "arguments[0].click();", next_page_button)
                                logger.info(f"Clicked review pagination button for page {page_count + 2} for {product_url}.")
                                await asyncio.to_thread(time.sleep, random.uniform(2, 3))
                            else:
                                logger.info(f"No more review pages or 'show more' button found for {product_url}.")
                                break 
                        except Exception as page_e:
                            logger.info(f"No more review pages or 'show more' (or error clicking pagination) for {product_url}: {page_e}")
                            break 
                    page_count += 1
                product_details["scrape_status"]["reviews"] = f"success_found_{len(product_details['reviews'])}"
            except TimeoutException:
                logger.warning(f"Timeout: Review tab or initial reviews not found for {product_url}. Context: {current_context}")
                product_details["scrape_status"]["reviews"] = f"failure_timeout_or_not_found: {current_context}"
            except Exception as e:
                logger.error(f"Error scraping reviews for {product_url}. Context: {current_context}. Error: {e}")
                product_details["scrape_status"]["reviews"] = f"failure_exception: {type(e).__name__} in {current_context}"
        else:
            logger.warning(f"Review selectors not fully defined for {product_url}. Skipping review scraping.")
            product_details["scrape_status"]["reviews"] = "skipped_selectors_undefined"

        # --- Q&A 스크래핑 ---
        current_context = "qna tab click"
        if QNA_TAB_SELECTOR and QNA_ITEM_SELECTOR:
            try:
                qna_tab_button = await asyncio.to_thread(
                    WebDriverWait(driver, 15).until,
                    EC.element_to_be_clickable((By.CSS_SELECTOR, QNA_TAB_SELECTOR))
                )
                await asyncio.to_thread(driver.execute_script, "arguments[0].click();", qna_tab_button)
                await asyncio.to_thread(time.sleep, random.uniform(2, 4))

                current_context = "qna scraping"
                page_count_qna = 0
                max_qna_pages = 5 

                while page_count_qna < max_qna_pages:
                    await asyncio.to_thread(driver.execute_script, "window.scrollTo(0, document.body.scrollHeight);")
                    await asyncio.to_thread(time.sleep, random.uniform(1, 2))
                    
                    soup_qna = BeautifulSoup(await asyncio.to_thread(lambda: driver.page_source), "html.parser")
                    qna_elements = soup_qna.select(QNA_ITEM_SELECTOR)
                    
                    current_qna_count = len(product_details["qna"])
                    for qna_el in qna_elements[current_qna_count:]:
                        question = qna_el.select_one(QNA_QUESTION_SELECTOR).text.strip() if qna_el.select_one(QNA_QUESTION_SELECTOR) else "N/A"
                        answer_el = qna_el.select_one(QNA_ANSWER_SELECTOR)
                        answer = answer_el.text.strip() if answer_el else "답변 대기중이거나 없음"
                        author = qna_el.select_one(QNA_AUTHOR_SELECTOR).text.strip() if qna_el.select_one(QNA_AUTHOR_SELECTOR) else "N/A"
                        date = qna_el.select_one(QNA_DATE_SELECTOR).text.strip() if qna_el.select_one(QNA_DATE_SELECTOR) else "N/A"
                        product_details["qna"].append({
                            "question": question, "answer": answer, "author": author, "date": date
                        })

                    try:
                        show_more_qna_button = await asyncio.to_thread(
                            WebDriverWait(driver, 5).until,
                            EC.element_to_be_clickable((By.CSS_SELECTOR, QNA_SHOW_MORE_BUTTON_SELECTOR))
                        )
                        await asyncio.to_thread(driver.execute_script, "arguments[0].click();", show_more_qna_button)
                        logger.info(f"Clicked Q&A 'show more' button for {product_url}. Page {page_count_qna + 1}")
                        await asyncio.to_thread(time.sleep, random.uniform(2, 3))
                    except TimeoutException:
                        try:
                            page_buttons_qna = await asyncio.to_thread(lambda: driver.find_elements(By.CSS_SELECTOR, QNA_PAGINATION_SELECTOR))
                            next_page_qna_button = next((btn for btn in page_buttons_qna if btn.text == str(page_count_qna + 2)), None)
                            if next_page_qna_button and next_page_qna_button.is_enabled():
                                await asyncio.to_thread(driver.execute_script, "arguments[0].click();", next_page_qna_button)
                                logger.info(f"Clicked Q&A pagination button for page {page_count_qna + 2} for {product_url}.")
                                await asyncio.to_thread(time.sleep, random.uniform(2, 3))
                            else:
                                logger.info(f"No more Q&A pages or 'show more' button found for {product_url}.")
                                break
                        except Exception as page_q_e:
                            logger.info(f"No more Q&A pages or 'show more' (or error clicking pagination) for {product_url}: {page_q_e}")
                            break
                    page_count_qna += 1
                product_details["scrape_status"]["qna"] = f"success_found_{len(product_details['qna'])}"
            except TimeoutException:
                logger.warning(f"Timeout: Q&A tab or initial Q&A items not found for {product_url}. Context: {current_context}")
                product_details["scrape_status"]["qna"] = f"failure_timeout_or_not_found: {current_context}"
            except Exception as e:
                logger.error(f"Error scraping Q&A for {product_url}. Context: {current_context}. Error: {e}")
                product_details["scrape_status"]["qna"] = f"failure_exception: {type(e).__name__} in {current_context}"
        else:
            logger.warning(f"Q&A selectors not fully defined for {product_url}. Skipping Q&A scraping.")
            product_details["scrape_status"]["qna"] = "skipped_selectors_undefined"

    except UnexpectedAlertPresentException as e:
        logger.error(f"Critical UnexpectedAlertPresentException during scraping {product_url} in context '{current_context}': {e.alert_text if e.alert_text else str(e)}")
        product_details["error"] = f"UnexpectedAlertPresentException in {current_context}: {e.alert_text if e.alert_text else str(e)}"
        product_details["scrape_status"]["critical"] = f"UnexpectedAlertPresentException in {current_context}"
        if driver:
            try:
                # 예외 발생 시 알림창을 닫으려고 시도
                alert = driver.switch_to.alert
                alert.accept()
            except Exception as alert_e:
                logger.error(f"Could not dismiss alert during critical error handling: {alert_e}")
    except Exception as e:
        logger.error(f"An error occurred while scraping {product_url} in context '{current_context}': {e}", exc_info=True)
        product_details["error"] = f"General error in {current_context}: {str(e)}"
        product_details["scrape_status"]["critical"] = f"General error in {current_context}: {type(e).__name__}"
    finally:
        if driver:
            await asyncio.to_thread(driver.quit)
            logger.info(f"WebDriver closed for product details page: {product_url}")

    if product_details["product_name"] == "N/A" and product_details["price"] == "N/A" and product_details["image_url"] == "N/A":
        logger.warning(f"Failed to scrape significant details for {product_url}")
        if "error" not in product_details:
             product_details["error"] = "Failed to scrape significant details (name, price, image)."
        product_details["scrape_status"]["overall"] = "failure_significant_details_missing"
    return product_details

async def get_coupan_report_for_bono_house(keyword: str, max_products: int = 3) -> List[Dict[str, Any]]:
    """
    주어진 키워드로 쿠팡에서 상품 정보를 스크랩하여 보고서 형태로 반환합니다.
    Parameters:
        keyword (str): The search term to use on Coupang.
        max_products (int): The maximum number of products to scrape details for from the search results.
                           (Note: `search_products_on_coupan` which would use this is currently not implemented)
    """
    report_data = []

    # The 'keyword' and 'max_products' parameters allow for configurable scraping.
    logger.info(f"Starting Coupang report generation for keyword: '{keyword}', max_products: {max_products}")
    
    # TODO: Implement `search_products_on_coupan(keyword, max_pages)` to get actual product URLs.
    # This function would likely involve:
    # 1. Navigating to Coupang's search page with the keyword.
    # 2. Parsing the search results to extract product page URLs.
    # 3. Handling pagination if `max_pages` > 1.
    # product_page_urls = await search_products_on_coupan(keyword, max_pages=1) 
    product_page_urls = [] # Using an empty list as `search_products_on_coupan` is not yet implemented.

    if not product_page_urls:
        logger.warning(f"No products found for keyword: '{keyword}' via search. Returning mock data as fallback.")
    else:
        logger.info(f"Will attempt to scrape details for up to {max_products} products for keyword '{keyword}'.")
        for i, product_url in enumerate(product_page_urls[:max_products]):
            logger.info(f"Scraping product {i+1}/{len(product_page_urls[:max_products])} for keyword '{keyword}': {product_url}")
            details = await scrape_product_details_from_coupan(product_url)
            if details.get("product_name") and details["product_name"] != "N/A": # 유효한 상품명이 있는 경우
                report_data.append(details)
            
            if i < len(product_page_urls[:max_products]) - 1:
                await asyncio.sleep(random.uniform(2.0, 4.0))

    if not report_data or not any(item.get("product_name") and item["product_name"] != "N/A" for item in report_data):
        logger.warning(f"No actual data scraped or data was insufficient for '{keyword}'. Using MOCK DATA as fallback.")
        # Mock data uses a generic keyword in product names to show it's mock
        mock_keyword_display = keyword if keyword else "보노 하우스" # Fallback for mock display if keyword is empty
        report_data = [
            {
                "product_name": f"[{mock_keyword_display}] 따뜻한 극세사 겨울 침구세트 (Q) (Mock)",
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
