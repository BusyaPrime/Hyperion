package hyperion.sdk;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Java SDK клиент для HYPERION сервисов.
 *
 * Высокоуровневый API: сабмитим джобы, мониторим статус, забираем результаты.
 */
public class HyperionClient implements AutoCloseable {
    private static final Logger log = LoggerFactory.getLogger(HyperionClient.class);

    private final ManagedChannel inferenceChannel;
    private final ManagedChannel orchestratorChannel;

    public HyperionClient(String inferenceHost, int inferencePort,
                          String orchestratorHost, int orchestratorPort) {
        this.inferenceChannel = ManagedChannelBuilder
                .forAddress(inferenceHost, inferencePort)
                .usePlaintext()
                .build();
        this.orchestratorChannel = ManagedChannelBuilder
                .forAddress(orchestratorHost, orchestratorPort)
                .usePlaintext()
                .build();

        log.info("HyperionClient connected: inference={}:{}, orchestrator={}:{}",
                inferenceHost, inferencePort, orchestratorHost, orchestratorPort);
    }

    public HyperionClient(String host, int inferencePort, int orchestratorPort) {
        this(host, inferencePort, host, orchestratorPort);
    }

    /**
     * Сабмитим инференс-джобу в оркестратор.
     *
     * @param modelSource Исходник модели.
     * @param config      Конфиг инференса.
     * @param priority    Приоритет (выше = срочнее).
     * @return Job ID.
     */
    public String submitJob(String modelSource, Map<String, Object> config, int priority) {
        // В проде: используем сгенерированный gRPC stub
        // OrchestratorServiceGrpc.OrchestratorServiceBlockingStub stub = ...
        // SubmitJobRequest request = SubmitJobRequest.newBuilder()...build();
        // SubmitJobResponse response = stub.submitJob(request);
        log.info("Submitting job: model={}, priority={}", modelSource.length(), priority);
        return "placeholder-job-id";
    }

    /**
     * Получаем статус джобы.
     *
     * @param jobId Job ID.
     * @return Строка статуса.
     */
    public String getJobStatus(String jobId) {
        log.info("Getting status for job: {}", jobId);
        return "PENDING";
    }

    /**
     * Ждём завершения джобы, поллим с заданным интервалом.
     *
     * @param jobId          Job ID.
     * @param pollIntervalMs Интервал поллинга в мс.
     * @param timeoutMs      Максимальное время ожидания в мс.
     * @return Финальный статус.
     * @throws InterruptedException Если поток прервали.
     */
    public String waitForCompletion(String jobId, long pollIntervalMs, long timeoutMs)
            throws InterruptedException {
        long start = System.currentTimeMillis();
        while (System.currentTimeMillis() - start < timeoutMs) {
            String status = getJobStatus(jobId);
            if ("COMPLETED".equals(status) || "FAILED".equals(status) || "CANCELLED".equals(status)) {
                return status;
            }
            Thread.sleep(pollIntervalMs);
        }
        return "TIMEOUT";
    }

    /**
     * Отменяем запущенную джобу.
     *
     * @param jobId Job ID.
     * @return True если отменили успешно.
     */
    public boolean cancelJob(String jobId) {
        log.info("Cancelling job: {}", jobId);
        return true;
    }

    @Override
    public void close() {
        try {
            inferenceChannel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
            orchestratorChannel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            inferenceChannel.shutdownNow();
            orchestratorChannel.shutdownNow();
        }
        log.info("HyperionClient closed");
    }
}
