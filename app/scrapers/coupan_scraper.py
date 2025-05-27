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
REVIEW_TAB_SELECTOR = "a[name='review']" # Selector for the 'Reviews' tab.
QNA_TAB_SELECTOR = "a[name='productInquiry']"    # 예: <a name="productInquiry" href="...">문의</a>

# 리뷰 관련 선택자
REVIEW_CONTAINER_SELECTOR = "div.sdp-review__article__list" # A div that likely wraps multiple review articles.
REVIEW_ITEM_SELECTOR = "article.sdp-review__article__list" # Selector for an individual review entry/article. Each review seems to be an <article> with this class.
REVIEW_AUTHOR_SELECTOR = "span.sdp-review__article__list__info__user__name" # Selector for the author's name.
REVIEW_RATING_SELECTOR = "div.sdp-review__article__list__info__product-info__star-gray > span" # Selector for the star rating span (width style indicates rating).
REVIEW_DATE_SELECTOR = "div.sdp-review__article__list__info__product-info__reg-date" # Selector for the review date.
REVIEW_CONTENT_SELECTOR = "div.sdp-review__article__list__review__content" # Selector for the main text content of the review.
REVIEW_SHOW_MORE_BUTTON_SELECTOR = "button.sdp-review__article__page__more__button" # Selector for the button to load more review items/pages.
REVIEW_PAGINATION_SELECTOR = "button.sdp-review__article__page__num" # Selector for pagination buttons (page numbers).

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
        
        stabilization_time = random.uniform(3, 5) 
        logger.info(f"Waiting for page stabilization for {stabilization_time:.2f} seconds. URL: {product_url}")
        await asyncio.to_thread(time.sleep, stabilization_time)
        
        # 상품 기본 정보 스크래핑
        current_context = "product name scraping"
        try:
            product_name_element = await asyncio.to_thread(
                WebDriverWait(driver, 20).until, 
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
                    WebDriverWait(driver, 20).until, 
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
                        product_details["price"] = price_text_to_parse 
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
                    WebDriverWait(driver, 10).until, 
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
        if REVIEW_TAB_SELECTOR and REVIEW_ITEM_SELECTOR and REVIEW_CONTAINER_SELECTOR:
            try:
                review_tab_button = await asyncio.to_thread(
                    WebDriverWait(driver, 30).until, 
                    EC.element_to_be_clickable((By.CSS_SELECTOR, REVIEW_TAB_SELECTOR))
                )
                await asyncio.to_thread(driver.execute_script, "arguments[0].click();", review_tab_button)
                logger.info(f"Clicked review tab for {product_url}. Waiting for reviews to load...")
                await asyncio.to_thread(time.sleep, random.uniform(3.0, 6.0)) 

                await asyncio.to_thread(
                    WebDriverWait(driver, 25).until, 
                    EC.presence_of_element_located((By.CSS_SELECTOR, REVIEW_CONTAINER_SELECTOR))
                )
                logger.info(f"Review container/items initially present for {product_url}.")
                current_context = "reviews scraping"
                
                max_show_more_clicks = 25 
                
                for i in range(max_show_more_clicks):
                    try:
                        show_more_button = await asyncio.to_thread(
                            WebDriverWait(driver, 7).until, 
                            EC.element_to_be_clickable((By.CSS_SELECTOR, REVIEW_SHOW_MORE_BUTTON_SELECTOR))
                        )
                        await asyncio.to_thread(driver.execute_script, "arguments[0].click();", show_more_button)
                        logger.info(f"Clicked review 'show more' button ({i+1}/{max_show_more_clicks}) for {product_url}.")
                        await asyncio.to_thread(time.sleep, random.uniform(2.5, 5.5)) 
                    except TimeoutException:
                        logger.info(f"'Show more' reviews button not found or no longer clickable after {i} clicks for {product_url}.")
                        break
                    except Exception as e_sm:
                        logger.warning(f"Error clicking 'show more' reviews button for {product_url}: {e_sm}")
                        break
                
                max_pagination_clicks = 15 
                for i in range(max_pagination_clicks):
                    try:
                        page_buttons_non_active = await asyncio.to_thread(
                            driver.find_elements, By.CSS_SELECTOR, f"{REVIEW_PAGINATION_SELECTOR}:not([disabled]):not(.on)"
                        )
                        
                        clicked_next_page = False
                        current_page_elements = await asyncio.to_thread(driver.find_elements, By.CSS_SELECTOR, f"{REVIEW_PAGINATION_SELECTOR}.on")
                        current_page_num = 0
                        if current_page_elements and current_page_elements[0].text.isdigit():
                            current_page_num = int(current_page_elements[0].text)
                        
                        target_page_num = current_page_num + 1
                        next_page_button_found = None

                        for btn in page_buttons_non_active:
                            btn_text = await asyncio.to_thread(getattr, btn, 'text')
                            if btn_text.isdigit() and int(btn_text) == target_page_num:
                                next_page_button_found = btn
                                break
                        
                        if not next_page_button_found and page_buttons_non_active: 
                             next_page_button_found = page_buttons_non_active[0]


                        if next_page_button_found:
                            button_text_for_log = await asyncio.to_thread(getattr, next_page_button_found, 'text')
                            await asyncio.to_thread(driver.execute_script, "arguments[0].click();", next_page_button_found)
                            logger.info(f"Clicked review pagination button for page '{button_text_for_log}' ({i+1}/{max_pagination_clicks}) for {product_url}.")
                            await asyncio.to_thread(time.sleep, random.uniform(2.5, 5.5)) 
                            clicked_next_page = True
                        
                        if not clicked_next_page:
                            logger.info(f"No further active review pagination buttons found after {i} clicks for {product_url}.")
                            break
                    except Exception as e_page: 
                        logger.warning(f"Error clicking review pagination for {product_url}: {e_page}")
                        break

                logger.info(f"Finished loading attempts for reviews on {product_url}. Parsing all loaded reviews.")
                final_page_source = await asyncio.to_thread(getattr, driver, 'page_source')
                soup = BeautifulSoup(final_page_source, "html.parser")
                review_elements_final = soup.select(REVIEW_ITEM_SELECTOR)
                
                processed_reviews = set() 

                for review_el in review_elements_final:
                    try:
                        author = review_el.select_one(REVIEW_AUTHOR_SELECTOR).text.strip() if review_el.select_one(REVIEW_AUTHOR_SELECTOR) else "N/A"
                        rating_text = "0" 
                        rating_el = review_el.select_one(REVIEW_RATING_SELECTOR)
                        if rating_el and 'style' in rating_el.attrs:
                            style_attr = rating_el['style']
                            if "width:" in style_attr:
                                rating_text = style_attr.split("width:")[1].split("%")[0].strip()
                        rating = int(float(rating_text) / 20) if rating_text.replace('.', '', 1).isdigit() else 0
                        
                        date = review_el.select_one(REVIEW_DATE_SELECTOR).text.strip() if review_el.select_one(REVIEW_DATE_SELECTOR) else "N/A"
                        content_el = review_el.select_one(REVIEW_CONTENT_SELECTOR)
                        content = content_el.text.strip() if content_el else "N/A"
                        
                        review_tuple = (author, rating, date, content)
                        if review_tuple not in processed_reviews:
                            product_details["reviews"].append({
                                "author": author, "rating": rating, "date": date, "content": content
                            })
                            processed_reviews.add(review_tuple)
                    except (AttributeError, TypeError, ValueError, IndexError) as e_parse: 
                        logger.warning(f"Error parsing a single review item for {product_url}. Error: {e_parse}. Item HTML (snippet): {str(review_el)[:200]}")
                
                if product_details["reviews"]:
                    product_details["scrape_status"]["reviews"] = f"success_found_{len(product_details['reviews'])}"
                else:
                    product_details["scrape_status"]["reviews"] = "warning_no_reviews_extracted_after_load_attempts"

            except TimeoutException:
                logger.warning(f"Timeout: Review tab or initial review container not found for {product_url} using selectors '{REVIEW_TAB_SELECTOR}' / '{REVIEW_CONTAINER_SELECTOR}'. Context: {current_context}")
                product_details["scrape_status"]["reviews"] = f"failure_critical_timeout_or_not_found: {current_context}"
            except Exception as e:
                logger.error(f"Critical error during review scraping setup for {product_url}. Context: {current_context}. Error: {e}", exc_info=True)
                product_details["scrape_status"]["reviews"] = f"failure_critical_exception: {type(e).__name__} in {current_context}"
        else:
            logger.warning(f"Review selectors (tab, item, or container) not fully defined for {product_url}. Skipping review scraping.")
            product_details["scrape_status"]["reviews"] = "skipped_selectors_undefined"

        # --- Q&A 스크래핑 ---
        # TODO: Apply similar robust loading logic (increased waits, show more/pagination loop) for Q&A if needed.
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
    logger.info(f"Starting Coupang report generation for keyword: '{keyword}', max_products: {max_products}")
    
    # TODO: Implement `search_products_on_coupan(keyword, max_pages)` to get actual product URLs.
    product_page_urls = [] 

    if not product_page_urls:
        logger.warning(f"No products found for keyword: '{keyword}' via search. Returning mock data as fallback.")
    else:
        logger.info(f"Will attempt to scrape details for up to {max_products} products for keyword '{keyword}'.")
        for i, product_url in enumerate(product_page_urls[:max_products]):
            logger.info(f"Scraping product {i+1}/{len(product_page_urls[:max_products])} for keyword '{keyword}': {product_url}")
            details = await scrape_product_details_from_coupan(product_url)
            if details.get("product_name") and details["product_name"] != "N/A": 
                report_data.append(details)
            
            if i < len(product_page_urls[:max_products]) - 1:
                await asyncio.sleep(random.uniform(2.0, 4.0))

    if not report_data or not any(item.get("product_name") and item["product_name"] != "N/A" for item in report_data):
        logger.warning(f"No actual data scraped or data was insufficient for '{keyword}'. Using MOCK DATA as fallback.")
        mock_keyword_display = keyword if keyword else "보노 하우스" 
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

async def scrape_reviews_for_url(product_url: str) -> Dict[str, Any]:
    """
    Scrapes only the reviews for a given Coupang product URL.
    This function wraps `scrape_product_details_from_coupan` and extracts review-specific data.
    """
    logger.info(f"Attempting to scrape reviews specifically for URL: {product_url}")
    
    product_details = await scrape_product_details_from_coupan(product_url)
    
    reviews = product_details.get("reviews", [])
    scrape_status_dict = product_details.get("scrape_status", {})
    review_status = scrape_status_dict.get("reviews", "unknown status") 
    
    product_name = product_details.get("product_name", "N/A")
    general_error = product_details.get("error")

    if general_error:
        logger.error(f"Review scraping for {product_url} (Product: {product_name}) failed due to general error: {general_error}. Review-specific status was: {review_status}")
    elif isinstance(review_status, str) and review_status.startswith("success"):
        if reviews:
            logger.info(f"Successfully extracted {len(reviews)} reviews for {product_url} (Product: {product_name}). Status: {review_status}")
        else: 
            logger.info(f"Review scraping process for {product_url} (Product: {product_name}) was successful, but no reviews were found. Status: {review_status}")
    elif isinstance(review_status, str) and ("warning_no_reviews_extracted" in review_status or "skipped" in review_status):
        logger.info(f"Review scraping for {product_url} (Product: {product_name}) completed with status: {review_status}. No reviews extracted or process skipped.")
    else: 
        logger.warning(f"Review scraping for {product_url} (Product: {product_name}) completed with non-success status: {review_status}")
        
    return {
        "product_name": product_name,
        "reviews": reviews,
        "status": review_status, 
        "error": general_error   
    }

# === 테스트 코드 시작 (이전 테스트 코드가 있다면 이 블록으로 교체해주세요) ===
if __name__ == "__main__":
    import asyncio
    
    if not logging.getLogger().hasHandlers(): 
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    async def main_test_single_item(): 
        test_url = "https://www.coupang.com/vp/products/2287107738?itemId=3931285857&vendorItemId=71911563636" # Specific URL for this task
        
        print("--- Coupang Review Scraping Test ---")
        # Truncate URL for display if it's too long - no longer needed with this specific short URL
        print(f"Product URL: {test_url}")
        
        logger.info(f"단일 상품 리뷰 스크래핑 테스트 시작. URL: {test_url}") # Keep original Korean log
        details = await scrape_reviews_for_url(test_url)
        
        print(f"Product Name: {details.get('product_name', 'N/A')}")
        print(f"Review Scraping Status: {details.get('status', 'N/A')}")
        print(f"General Error: {details.get('error', 'None')}")
        
        reviews_data = details.get('reviews', [])
        print(f"Number of Reviews Scraped: {len(reviews_data)}\n")
            
        if reviews_data:
            print("First 5 Reviews:")
            for i, review in enumerate(reviews_data[:5]):
                print(f"{i+1}. Author: {review.get('author', 'N/A')}")
                print(f"   Rating: {review.get('rating', 'N/A')}/5")
                print(f"   Date: {review.get('date', 'N/A')}")
                content_snippet = review.get('content', 'N/A')
                content_snippet = content_snippet[:100] + "..." if len(content_snippet) > 100 else content_snippet
                print(f"   Content: {content_snippet}")
                print("   ---")
            if len(reviews_data) > 5:
                print(f"... and {len(reviews_data) - 5} more reviews.")
        else:
            print("No reviews found or extracted.")
        print("------------------------------------")

    asyncio.run(main_test_single_item())
# === 테스트 코드 끝 ===
