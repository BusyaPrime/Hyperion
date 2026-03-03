package hyperion.orchestrator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;

// JobScheduler — планировщик задач с приоритетами, ретраями и вотчдогом
public class JobScheduler {
    private static final Logger log = LoggerFactory.getLogger(JobScheduler.class);

    private final ExecutorService executor;
    private final PriorityBlockingQueue<InferenceJob> jobQueue;
    private final Map<String, InferenceJob> jobRegistry;
    private final int maxConcurrency;
    private final ScheduledExecutorService watchdog;

    public JobScheduler(int maxConcurrency, int queueCapacity) {
        this.maxConcurrency = maxConcurrency;
        this.executor = Executors.newFixedThreadPool(maxConcurrency);
        this.jobQueue = new PriorityBlockingQueue<>(queueCapacity,
                Comparator.comparingInt(InferenceJob::getPriority).reversed());
        this.jobRegistry = new ConcurrentHashMap<>();
        this.watchdog = Executors.newSingleThreadScheduledExecutor();

        startWatchdog();
    }

    public InferenceJob submitJob(InferenceJob job) {
        jobRegistry.put(job.getJobId(), job);
        jobQueue.add(job);

        log.info("Job submitted: id={}, priority={}", job.getJobId(), job.getPriority());

        processQueue();
        return job;
    }

    public Optional<InferenceJob> getJob(String jobId) {
        return Optional.ofNullable(jobRegistry.get(jobId));
    }

    public boolean cancelJob(String jobId) {
        InferenceJob job = jobRegistry.get(jobId);
        if (job == null) return false;

        if (job.getStatus() == InferenceJob.Status.PENDING ||
            job.getStatus() == InferenceJob.Status.RUNNING) {
            job.setStatus(InferenceJob.Status.CANCELLED);
            job.setCompletedAt(Instant.now());
            log.info("Job cancelled: id={}", jobId);
            return true;
        }
        return false;
    }

    public List<InferenceJob> listJobs(InferenceJob.Status statusFilter, int limit, int offset) {
        List<InferenceJob> result = new ArrayList<>();
        for (InferenceJob job : jobRegistry.values()) {
            if (statusFilter == null || job.getStatus() == statusFilter) {
                result.add(job);
            }
        }
        result.sort(Comparator.comparing(InferenceJob::getCreatedAt).reversed());

        int from = Math.min(offset, result.size());
        int to = Math.min(offset + limit, result.size());
        return result.subList(from, to);
    }

    public int getTotalCount(InferenceJob.Status statusFilter) {
        if (statusFilter == null) return jobRegistry.size();
        return (int) jobRegistry.values().stream()
                .filter(j -> j.getStatus() == statusFilter)
                .count();
    }

    private void processQueue() {
        while (!jobQueue.isEmpty()) {
            InferenceJob job = jobQueue.poll();
            if (job == null || job.getStatus() == InferenceJob.Status.CANCELLED) continue;

            executor.submit(() -> executeJob(job));
        }
    }

    private void executeJob(InferenceJob job) {
        job.setStatus(InferenceJob.Status.RUNNING);
        job.setStartedAt(Instant.now());
        log.info("Job started: id={}", job.getJobId());

        try {
            // В проде: вызываем Python gRPC InferenceService
            // Пока симулируем выполнение
            boolean success = callPythonInference(job);

            if (success) {
                job.setStatus(InferenceJob.Status.COMPLETED);
                job.setCompletedAt(Instant.now());
                log.info("Job completed: id={}, elapsed={}s",
                        job.getJobId(),
                        Duration.between(job.getStartedAt(), job.getCompletedAt()).getSeconds());
            } else {
                throw new RuntimeException("Inference returned failure");
            }

        } catch (Exception e) {
            log.error("Job failed: id={}, error={}", job.getJobId(), e.getMessage());

            // Если задача упала 3 раза — хватит мучить, переводим в FAILED
            if (job.getRetryCount() < job.getMaxRetries()) {
                job.incrementRetryCount();
                job.setStatus(InferenceJob.Status.PENDING);
                long backoffMs = (long) Math.pow(2, job.getRetryCount()) * 1000;
                log.info("Retrying job {} in {}ms (attempt {})",
                        job.getJobId(), backoffMs, job.getRetryCount());

                watchdog.schedule(() -> {
                    jobQueue.add(job);
                    processQueue();
                }, backoffMs, TimeUnit.MILLISECONDS);
            } else {
                job.setStatus(InferenceJob.Status.FAILED);
                job.setErrorMessage(e.getMessage());
                job.setCompletedAt(Instant.now());
            }
        }
    }

    private boolean callPythonInference(InferenceJob job) {
        // Placeholder: интеграция с Python gRPC клиентом
        // ManagedChannel channel = ManagedChannelBuilder.forAddress("localhost", 50051)
        //         .usePlaintext().build();
        // InferenceServiceGrpc.InferenceServiceBlockingStub stub = ...
        log.info("Calling Python inference for job {}", job.getJobId());

        // Проверяем таймаут
        if (job.getTimeoutSeconds() > 0) {
            long elapsed = Duration.between(job.getStartedAt(), Instant.now()).getSeconds();
            if (elapsed > job.getTimeoutSeconds()) {
                throw new RuntimeException("Job timed out after " + elapsed + "s");
            }
        }

        return true;
    }

    private void startWatchdog() {
        // Вотчдог — каждые 10 сек проверяем, не зависла ли задача
        watchdog.scheduleAtFixedRate(() -> {
            for (InferenceJob job : jobRegistry.values()) {
                if (job.getStatus() == InferenceJob.Status.RUNNING &&
                    job.getTimeoutSeconds() > 0 &&
                    job.getStartedAt() != null) {
                    long elapsed = Duration.between(job.getStartedAt(), Instant.now()).getSeconds();
                    if (elapsed > job.getTimeoutSeconds()) {
                        log.warn("Job {} timed out after {}s", job.getJobId(), elapsed);
                        job.setStatus(InferenceJob.Status.FAILED);
                        job.setErrorMessage("Timed out after " + elapsed + "s");
                        job.setCompletedAt(Instant.now());
                    }
                }
            }
        }, 10, 10, TimeUnit.SECONDS);
    }

    public void shutdown() {
        executor.shutdown();
        watchdog.shutdown();
        try {
            executor.awaitTermination(30, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
