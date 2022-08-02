package org.nadeemlab.impartial;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import static java.util.concurrent.TimeUnit.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.json.JSONArray;
import org.json.JSONObject;
import org.scijava.app.StatusService;


public class TrainProgress {
    private final StatusService status;
    private final MonaiLabelClient monaiLabel;
    private final ScheduledExecutorService scheduler =
            Executors.newScheduledThreadPool(1);

    public TrainProgress(final StatusService status, final MonaiLabelClient monaiLabel) {
        this.status = status;
        this.monaiLabel = monaiLabel;
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

    public void monitorTraining() {
        final Runnable beeper = new Runnable() {

            public void run() {
                final JSONObject trainConfig = monaiLabel.getInfo()
                        .getJSONObject("trainers")
                        .getJSONObject("impartial")
                        .getJSONObject("config");

                final int maxEpochs = trainConfig.getInt("max_epochs");

                final JSONObject jsonProgress = monaiLabel.getTrain();

                JSONArray jsonDetails = jsonProgress.getJSONArray("details");
                List<String> details = new ArrayList<String>();
                for (int i = 0; i < jsonDetails.length(); i++) {
                    details.add(jsonDetails.getString(i));
                }

                String[] epochKeywords = {"SupervisedTrainer", "Epoch:"};
                int lastEpoch = details.stream()
                        .filter(r -> containsWords(r, epochKeywords))
                        .map(TrainProgress::epochFromLog)
                        .reduce((first, second) -> second)
                        .orElse(0);

                String message = lastEpoch > 0 ? "Epoch: " + lastEpoch : "Initializing...";
                status.showStatus(lastEpoch, maxEpochs, message);

                if (Objects.equals(jsonProgress.getString("status"), "DONE")) {
                    status.showStatus(maxEpochs, maxEpochs, "Training done after " + lastEpoch + " epochs");
                    throw new RuntimeException("Training done");
                }
            }
        };

        final ScheduledFuture<?> beeperHandle = scheduler.scheduleAtFixedRate(
                beeper, 1, 1, SECONDS
        );
    }
}