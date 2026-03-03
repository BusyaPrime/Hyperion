FROM gradle:8.5-jdk17 AS build

WORKDIR /app
COPY java/ java/
COPY proto/ proto/

WORKDIR /app/java
RUN gradle :orchestrator-service:build --no-daemon -x test

FROM eclipse-temurin:17-jre

WORKDIR /app
COPY --from=build /app/java/orchestrator-service/build/libs/*.jar app.jar

EXPOSE 50052

ENTRYPOINT ["java", "-jar", "app.jar"]
