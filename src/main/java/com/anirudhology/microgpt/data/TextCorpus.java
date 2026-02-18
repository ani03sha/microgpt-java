package com.anirudhology.microgpt.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class TextCorpus {

    private static final Logger LOG = LoggerFactory.getLogger(TextCorpus.class);
    private final Random random = new Random(42);

    public List<String> readCorpus(final String path) {
        final Path filePath = Paths.get(path);

        if (Files.notExists(filePath)) {
            downloadFileFromExternalPath(filePath);
        }

        try {
            // Read file and store names in a list
            final List<String> docs = Files.readAllLines(filePath)
                    .stream()
                    .map(String::trim)
                    .filter(doc -> !doc.isEmpty())
                    .collect(Collectors.toCollection(ArrayList::new));

            // Shuffle the docs randomly
            Collections.shuffle(docs, this.random);
            LOG.info("Total names read: {}", docs.size());
            return docs;
        } catch (IOException e) {
            LOG.error("Error reading corpus file due to: {}", e.getMessage());
            throw new UncheckedIOException("Failed to read corpus file", e);
        }
    }

    private void downloadFileFromExternalPath(Path targetPath) {
        LOG.info("Downloading corpus from external path to: {}", targetPath.toAbsolutePath());

        try (final HttpClient client = HttpClient.newHttpClient()) {
            final HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"))
                    .build();
            // Download the file directly to the specified path
            client.send(request, HttpResponse.BodyHandlers.ofFile(targetPath));
            LOG.info("Downloaded file from external path successfully");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Download interrupted", e);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to download file from external path", e);
        }
    }
}
