package hyperion.orchestrator;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;

// InferenceJob — единица работы: модель, конфиг, приоритет, ретраи, таймаут
public class InferenceJob {

    public enum Status {
        PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    }

    private final String jobId;
    private final String modelSource;
    private final Map<String, Object> config;
    private final int priority;
    private final int maxRetries;
    private final long timeoutSeconds;
    private final Map<String, String> labels;

    private Status status;
    private int retryCount;
    private Instant createdAt;
    private Instant startedAt;
    private Instant completedAt;
    private String errorMessage;
    private Object result;

    public InferenceJob(
            String modelSource,
            Map<String, Object> config,
            int priority,
            int maxRetries,
            long timeoutSeconds,
            Map<String, String> labels) {
        this.jobId = UUID.randomUUID().toString();
        this.modelSource = modelSource;
        this.config = config;
        this.priority = priority;
        this.maxRetries = maxRetries;
        this.timeoutSeconds = timeoutSeconds;
        this.labels = labels;
        this.status = Status.PENDING;
        this.retryCount = 0;
        this.createdAt = Instant.now();
    }

    public String getJobId() { return jobId; }
    public String getModelSource() { return modelSource; }
    public Map<String, Object> getConfig() { return config; }
    public int getPriority() { return priority; }
    public int getMaxRetries() { return maxRetries; }
    public long getTimeoutSeconds() { return timeoutSeconds; }
    public Map<String, String> getLabels() { return labels; }
    public Status getStatus() { return status; }
    public int getRetryCount() { return retryCount; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getStartedAt() { return startedAt; }
    public Instant getCompletedAt() { return completedAt; }
    public String getErrorMessage() { return errorMessage; }
    public Object getResult() { return result; }

    public void setStatus(Status status) { this.status = status; }
    public void setStartedAt(Instant startedAt) { this.startedAt = startedAt; }
    public void setCompletedAt(Instant completedAt) { this.completedAt = completedAt; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public void setResult(Object result) { this.result = result; }
    public void incrementRetryCount() { this.retryCount++; }
}
