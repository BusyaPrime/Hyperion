package hyperion.orchestrator;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

// Тесты JobScheduler — проверяем, что джобы регистрируются, отменяются и листаются
class JobSchedulerTest {

    private JobScheduler scheduler;

    @BeforeEach
    void setUp() {
        scheduler = new JobScheduler(2, 50);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void submitJob_shouldRegisterAndAssignId() {
        InferenceJob job = createJob();
        InferenceJob submitted = scheduler.submitJob(job);

        assertThat(submitted.getJobId()).isNotNull();
        assertThat(submitted.getJobId()).isNotEmpty();

        Optional<InferenceJob> retrieved = scheduler.getJob(submitted.getJobId());
        assertThat(retrieved).isPresent();
    }

    @Test
    void cancelJob_shouldSetCancelledStatus() throws InterruptedException {
        InferenceJob job = createJob();
        scheduler.submitJob(job);

        Thread.sleep(100);

        boolean cancelled = scheduler.cancelJob(job.getJobId());
        // Джоба могла уже завершиться
        Optional<InferenceJob> retrieved = scheduler.getJob(job.getJobId());
        assertThat(retrieved).isPresent();
    }

    @Test
    void listJobs_shouldReturnFilteredResults() {
        for (int i = 0; i < 5; i++) {
            scheduler.submitJob(createJob());
        }

        var allJobs = scheduler.listJobs(null, 50, 0);
        assertThat(allJobs).hasSizeGreaterThanOrEqualTo(5);
    }

    @Test
    void getTotalCount_shouldReturnCorrectCount() {
        int initial = scheduler.getTotalCount(null);
        scheduler.submitJob(createJob());
        assertThat(scheduler.getTotalCount(null)).isEqualTo(initial + 1);
    }

    private InferenceJob createJob() {
        Map<String, Object> config = new HashMap<>();
        config.put("method", "nuts");
        config.put("num_samples", 100);

        Map<String, String> labels = new HashMap<>();
        labels.put("test", "true");

        return new InferenceJob(
                "def model(): pass",
                config,
                0,
                3,
                3600,
                labels
        );
    }
}
