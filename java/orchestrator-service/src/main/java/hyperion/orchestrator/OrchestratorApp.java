package hyperion.orchestrator;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

// Главное приложение оркестратора — поднимаем gRPC, крутим JobScheduler
public class OrchestratorApp {
    private static final Logger log = LoggerFactory.getLogger(OrchestratorApp.class);

    private final int port;
    private Server server;
    private final JobScheduler scheduler;

    public OrchestratorApp(int port) {
        this.port = port;
        this.scheduler = new JobScheduler(4, 100);
    }

    public void start() throws IOException {
        server = ServerBuilder.forPort(port)
                .addService(new OrchestratorServiceImpl(scheduler))
                .build()
                .start();

        log.info("HYPERION Orchestrator started on port {}", port);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            log.info("Shutting down orchestrator...");
            OrchestratorApp.this.stop();
        }));
    }

    public void stop() {
        if (server != null) {
            scheduler.shutdown();
            server.shutdown();
        }
    }

    public void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        int port = 50052;
        if (args.length > 0) {
            port = Integer.parseInt(args[0]);
        }

        OrchestratorApp app = new OrchestratorApp(port);
        app.start();
        app.blockUntilShutdown();
    }
}
