package hyperion.orchestrator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Реализация gRPC OrchestratorService.
 *
 * Note: Используем plain Java типы как placeholder. В проде
 * наследовались бы от сгенерированного gRPC base class из hyperion.proto.
 */
public class OrchestratorServiceImpl {
    private static final Logger log = LoggerFactory.getLogger(OrchestratorServiceImpl.class);

    private final JobScheduler scheduler;

    public OrchestratorServiceImpl(JobScheduler scheduler) {
        this.scheduler = scheduler;
    }

    public Map<String, Object> submitJob(Map<String, Object> request) {
        String modelSource = (String) request.getOrDefault("model_source", "");
        @SuppressWarnings("unchecked")
        Map<String, Object> config = (Map<String, Object>) request.getOrDefault("config", new HashMap<>());
        int priority = (int) request.getOrDefault("priority", 0);
        int maxRetries = (int) request.getOrDefault("max_retries", 3);
        long timeoutSeconds = (long) request.getOrDefault("timeout_seconds", 3600L);
        @SuppressWarnings("unchecked")
        Map<String, String> labels = (Map<String, String>) request.getOrDefault("labels", new HashMap<>());

        InferenceJob job = new InferenceJob(
                modelSource, config, priority, maxRetries, timeoutSeconds, labels);
        scheduler.submitJob(job);

        Map<String, Object> response = new HashMap<>();
        response.put("job_id", job.getJobId());
        response.put("status", job.getStatus().name());
        response.put("queue_position", 0);
        return response;
    }

    public Map<String, Object> getJobStatus(String jobId) {
        Optional<InferenceJob> jobOpt = scheduler.getJob(jobId);
        Map<String, Object> response = new HashMap<>();

        if (jobOpt.isPresent()) {
            InferenceJob job = jobOpt.get();
            response.put("job_id", job.getJobId());
            response.put("status", job.getStatus().name());
            response.put("retries", job.getRetryCount());
            response.put("error_message", job.getErrorMessage());
            if (job.getStartedAt() != null) {
                response.put("started_at", job.getStartedAt().toEpochMilli());
            }
            if (job.getCompletedAt() != null) {
                response.put("completed_at", job.getCompletedAt().toEpochMilli());
            }
        } else {
            response.put("error", "Job not found: " + jobId);
        }

        return response;
    }

    public Map<String, Object> cancelJob(String jobId) {
        boolean cancelled = scheduler.cancelJob(jobId);
        Map<String, Object> response = new HashMap<>();
        response.put("job_id", jobId);
        response.put("cancelled", cancelled);
        return response;
    }

    public Map<String, Object> listJobs(Map<String, Object> request) {
        String statusStr = (String) request.get("status_filter");
        InferenceJob.Status statusFilter = null;
        if (statusStr != null && !statusStr.isEmpty()) {
            try {
                statusFilter = InferenceJob.Status.valueOf(statusStr);
            } catch (IllegalArgumentException ignored) {}
        }

        int limit = (int) request.getOrDefault("limit", 50);
        int offset = (int) request.getOrDefault("offset", 0);

        List<InferenceJob> jobs = scheduler.listJobs(statusFilter, limit, offset);
        int totalCount = scheduler.getTotalCount(statusFilter);

        Map<String, Object> response = new HashMap<>();
        response.put("total_count", totalCount);
        response.put("jobs", jobs.stream().map(j -> {
            Map<String, Object> m = new HashMap<>();
            m.put("job_id", j.getJobId());
            m.put("status", j.getStatus().name());
            m.put("retries", j.getRetryCount());
            return m;
        }).toList());

        return response;
    }
}
