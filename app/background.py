import asyncio
import logging
from datetime import datetime, timedelta
from fastapi import BackgroundTasks
from .cache import cache_report, get_cached_report

logger = logging.getLogger(__name__)
report_queue: list = []
is_processing = False

async def queue_report_generation(params, background_tasks: BackgroundTasks): # background_tasks will be unused if add_task is removed
    """보고서 생성 작업을 큐에 추가하고, 실제 처리는 스케줄러에 위임"""
    logger.info(f"QUEUE_REPORT_GENERATION_CALLED with params: {params}")
    
    # 중복 체크
    for i, queued_params in enumerate(report_queue):
        if (getattr(queued_params, 'from_date', None) == getattr(params, 'from_date', None) and
            getattr(queued_params, 'to_date', None) == getattr(params, 'to_date', None) and
            getattr(queued_params, 'item', None) == getattr(params, 'item', None) and
            getattr(queued_params, 'category', None) == getattr(params, 'category', None)):
            logger.info(f"QUEUE_REPORT_GENERATION: Duplicate report request. Params: {params} already in queue at position {i+1}.")
            return {"message": f"동일한 보고서가 이미 큐의 {i+1}번째에 있습니다.", "queue_position": i + 1}

    report_queue.append(params)
    logger.info(f"QUEUE_REPORT_GENERATION: Report params added to queue: {params}. Current queue size: {len(report_queue)}.")
    
    # background_tasks.add_task(process_report_queue) # REMOVED: Let scheduler handle processing.
    logger.info("QUEUE_REPORT_GENERATION: Task NOT added to background. Processing will be handled by scheduler.")
    
    return {"message": "보고서 생성이 예약되었습니다. 스케줄러에 의해 처리됩니다.", "queue_position": len(report_queue)}

async def process_report_queue():
    global is_processing
    logger.info(f"PROCESS_REPORT_QUEUE_CALLED. is_processing: {is_processing}, queue_size: {len(report_queue)}")
    if is_processing:
        logger.info("PROCESS_REPORT_QUEUE_EXITING: is_processing is True.")
        return
    
    if not report_queue:
        logger.info("PROCESS_REPORT_QUEUE_EXITING: report_queue is empty.")
        # is_processing = False # 큐가 비었을 때는 is_processing을 True로 바꾸지 않았으므로, 여기서 False로 할 필요 없음
        return

    is_processing = True
    logger.info("PROCESS_REPORT_QUEUE_PROCEEDING: Set is_processing=True.")
    try:
        while report_queue:
            params = report_queue.pop(0)
            logger.info(f"PROCESS_REPORT_QUEUE_PROCESSING_ITEM: {params}")
            cache_key = f"report_{params.from_date}_{params.to_date}_{params.item or ''}_{params.category or ''}"
            logger.info(f"백그라운드 보고서 생성 시작 (cache_key: {cache_key}): {params}")
            try:
                # 동적 import로 순환 참조 방지
                from .main import generate_shipment_report
                report_data = await generate_shipment_report(params)
                cache_report(cache_key, report_data)
                logger.info(f"보고서 생성 완료: {cache_key}")
            except Exception as e:
                logger.error(f"보고서 생성 실패: {e}")
    finally:
        is_processing = False
        logger.info("PROCESS_REPORT_QUEUE_FINISHED: Set is_processing=False.")

async def schedule_periodic_reports():
    """일별/주별/월별 보고서를 자동 생성"""
    yesterday = datetime.now() - timedelta(days=1)
    # TrendParams 동적 import로 순환 참조 방지
    from .main import TrendParams
    daily_params = TrendParams(from_date=yesterday.date(), to_date=yesterday.date())
    report_queue.append(daily_params)

    if datetime.now().weekday() == 0:
        last_week_end = datetime.now() - timedelta(days=1)
        last_week_start = last_week_end - timedelta(days=6)
        weekly_params = TrendParams(from_date=last_week_start.date(), to_date=last_week_end.date())
        report_queue.append(weekly_params)

    if datetime.now().day == 1:
        last_month = datetime.now().replace(day=1) - timedelta(days=1)
        last_month_start = last_month.replace(day=1)
        monthly_params = TrendParams(from_date=last_month_start.date(), to_date=last_month.date())
        report_queue.append(monthly_params)

    # await process_report_queue() # 큐에 항목 추가 후, 실제 처리는 별도의 스케줄러 작업으로 위임
    if report_queue:
        logger.info(f"보고서 큐에 작업이 추가되었습니다. 현재 큐 크기: {len(report_queue)}")
