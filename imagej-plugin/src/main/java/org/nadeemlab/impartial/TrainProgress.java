package org.nadeemlab.impartial;


import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.util.concurrent.TimeUnit.SECONDS;


public class TrainProgress {
    public TrainProgress() {
    }

    private static boolean containsWords(String input, String[] words) {
        return Arrays.stream(words).allMatch(input::contains);
    }

    private static int epochFromLog(String line) {
        String regex = "Epoch:\\s(.*?)\\/\\d+";
        Pattern p = Pattern.compile(regex);
        Matcher m = p.matcher(line);

        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    public static void monitorTraining(ImpartialController controller) {
        final Runnable beeper = () -> {

            final JSONObject jsonProgress = controller.getTrain();

            JSONArray jsonDetails = jsonProgress.getJSONArray("details");
            List<String> details = new ArrayList<String>();
            for (int i = 0; i < jsonDetails.length(); i++) {
                details.add(jsonDetails.getString(i));
            }

            String[] epochKeywords = {"Epoch:", "train_loss:"};
            int lastEpoch = details.stream()
                    .filter(r -> containsWords(r, epochKeywords))
                    .map(TrainProgress::epochFromLog)
                    .reduce((first, second) -> second)
                    .orElse(0);

            final int maxEpochs = controller.getMaxEpochs();
            String message = lastEpoch > 0 ? "Epoch: " + lastEpoch : "Initializing...";
            controller.showStatus(lastEpoch, maxEpochs , message);

            if (Objects.equals(jsonProgress.getString("status"), "DONE")) {
                controller.showStatus(maxEpochs, maxEpochs, "Training done after " + lastEpoch + " epochs");
                throw new RuntimeException("Training done");
            }
        };

        final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        final ScheduledFuture<?> beeperHandle = scheduler.scheduleAtFixedRate(
                beeper, 1, 1, SECONDS
        );
    }
}