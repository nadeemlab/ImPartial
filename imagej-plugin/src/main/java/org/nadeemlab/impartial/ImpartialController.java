package org.nadeemlab.impartial;

import ij.ImagePlus;
import ij.gui.*;
import ij.io.Opener;
import ij.plugin.ContrastEnhancer;
import ij.plugin.LutLoader;
import ij.plugin.frame.RoiManager;
import ij.process.FloatProcessor;
import net.imagej.ops.OpService;
import net.imagej.ops.geom.geom2d.LabelRegionToPolygonConverter;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.binary.Thresholder;
import net.imglib2.algorithm.labeling.ConnectedComponents;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.roi.geom.real.Polygon2D;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegion;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.type.logic.BitType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.FloatType;
import org.json.JSONArray;
import org.json.JSONObject;
import org.scijava.Context;
import org.scijava.app.StatusService;
import org.scijava.plugin.Parameter;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.image.IndexColorModel;
import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Timer;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class ImpartialController {
    final JFileChooser fileChooser = new JFileChooser();
    private final JFrame mainFrame = new JFrame("ImPartial");
    private final ImpartialContentPane contentPane = new ImpartialContentPane(this);
    private MonaiLabelClient monaiClient;
    private final SessionClient sessionClient = new SessionClient(Config.API_URL);
    private final Hashtable<String, ModelOutput> modelOutputs = new Hashtable<>();
    private final CapacityProvider capacityProvider = new CapacityProvider(this);
    private final RestoreSessionTask restoreSessionTask = new RestoreSessionTask(this);
    private final ImageUploader imageUploader = new ImageUploader(this);
    private final IndexColorModel redGreenLut = LutLoader.getLut("redgreen");
    private final Timer timer = new Timer();
    LabelRegionToPolygonConverter regionToPolygonConverter = new LabelRegionToPolygonConverter();
    private ImageWindow imageWindow;
    private File imageFile;
    private File labelFile;
    private int currentEpoch = 0;
    private TimerTask endOfSessionWarningTask;
    private SwingWorker<Void, Void> trainWorker;
    private String sessionId = null;
    private int numberOfChannels = 0;
    private final Properties props = new Properties();
    @Parameter
    private OpService ops;
    @Parameter
    private StatusService status;

    public ImpartialController(final Context context) {
        context.inject(this);
        context.inject(regionToPolygonConverter);
        createTempFiles();

        fileChooser.setMultiSelectionEnabled(true);

        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setResizable(false);

        //Create and set up the content pane.
        contentPane.setOpaque(true); //content panes must be opaque
        mainFrame.setContentPane(contentPane);

        mainFrame.pack();
        mainFrame.setVisible(true);
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

    private void showIOError(IOException e) {
        if (e instanceof InterruptedIOException)
            return;

        JOptionPane.showMessageDialog(contentPane,
                e.getMessage(),
                e.getClass().getName(),
                JOptionPane.ERROR_MESSAGE
        );
    }

    private void createTempFiles() {
        /*
         * This temporary files are used to store locally the current image,
         * and a zip file with rois for the label.
         * */
        try {
            imageFile = File.createTempFile("impartial-image-", ".png");
            imageFile.deleteOnExit();

            labelFile = File.createTempFile("impartial-label-", ".zip");
            labelFile.deleteOnExit();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public JPanel getContentPane() {
        return this.contentPane;
    }

    public void start() {
        if (contentPane.getRequestServerCheckBox()) {
            capacityProvider.provisionServer();
        } else {
            try {
                monaiClient = new MonaiLabelClient(contentPane.getUrl());
                monaiClient.getInfo();
            } catch (IOException e) {
                onStopped();
                showIOError(e);
                return;
            }
            onStarted();
        }
    }

    public void onStarted() {
        if (contentPane.getRequestServerCheckBox()) {
            contentPane.setSession(sessionId);
        }
        numberOfChannels = getNumberOfChannels();
        contentPane.onStarted();
    }

    public void stop() {
        if (contentPane.getRequestServerCheckBox()) {
            try {
                sessionClient.stopSession();
            } catch (IOException e) {
                showIOError(e);
            }
        }
        onStopped();
    }

    public void onStopped() {
        contentPane.onStopped();
        modelOutputs.clear();
        sessionId = null;
        numberOfChannels = 0;

        if (imageWindow != null) {
            imageWindow.setVisible(false);
        }
    }

    public void restoreSession(String sessionId) throws IOException {
        try {
            sessionClient.restoreSession(sessionId);
        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }

    public void updateImage() {
        try {
            clearRoiManager();
            String imageId = contentPane.getSelectedImageId();

            displayImage(imageId);

            JSONObject imageInfo = getImageInfo(imageId);
            boolean hasLabel = imageInfo.getJSONObject("labels").length() > 0;

            contentPane.setEnabledLabel(hasLabel);
            contentPane.setEnabledSubmit(hasLabel && contentPane.getSelectedViews().contains("Label"));
            if (!hasLabel)
                contentPane.setSelectedLabel(false);

            contentPane.setEnabledInfer(true);
            contentPane.setEnabledInferAndEntropy(modelOutputs.containsKey(imageId));

            updateDisplay();

        } catch (IOException e) {
            showIOError(e);
        }
    }

    public void displayImage(String imageId) {
        final ImagePlus imp = getImage(imageId);
        if (imageWindow == null) {
            imageWindow = new ImageWindow(imp);
            imageWindow.addComponentListener(new ComponentAdapter() {
                public void componentResized(ComponentEvent componentEvent) {
                    resetLayout();
                }
            });
            resetLayout();
        } else {
            imageWindow.setImage(imp);
            imageWindow.pack();
        }

        if (!imageWindow.isVisible()) {
            imageWindow.setVisible(true);
            resetLayout();
        }
    }

    private ImagePlus getImage(String imageId) {
        try {
            byte[] imageBytes = monaiClient.getDatastoreImage(imageId);

            FileOutputStream stream = new FileOutputStream(imageFile.getAbsolutePath());
            stream.write(imageBytes);
            stream.close();
        } catch (IOException e) {
            showIOError(e);
        }

        return new Opener().openImage(imageFile.getAbsolutePath());
    }

    public JSONObject getDatastore() {
        try {
            JSONObject datastore = monaiClient.getDatastore();
            JSONObject images = datastore.getJSONObject("objects");

            String[] keys = JSONObject.getNames(images);
            if (keys != null) {
                for (String key : keys) {
                    if (!monaiClient.headDatastoreImage(key)) {
                        images.remove(key);
                    }
                }
            }

            return datastore;

        } catch (IOException e) {
            showIOError(e);
            return new JSONObject();
        }
    }

    public int getNumberOfChannels() {
        JSONObject images = getImages();
        if (images.length() == 0)
            return 0;
        String first_image_id = JSONObject.getNames(images)[0];
        ImagePlus imp = getImage(first_image_id);

        return getNumberOfChannels(imp);
    }

    private int getNumberOfChannels(ImagePlus imp) {
        if (imp.isComposite())
            return imp.getNChannels();
        else
            return imp.getProcessor().getNChannels();
    }

    public void displayLabel() {
        try {
            String imageId = contentPane.getSelectedImageId();
            byte[] labelBytes = monaiClient.getDatastoreLabel(imageId);

            FileOutputStream stream = new FileOutputStream(labelFile);
            stream.write(labelBytes);
            stream.close();

            RoiManager roiManager = RoiManager.getRoiManager();

            JSONObject imageInfo = getImageInfo(imageId);
            if (imageInfo.getJSONObject("labels").length() > 0) {
                roiManager.runCommand("Open", labelFile.getAbsolutePath());
                roiManager.runCommand("Show All");
            }

        } catch (IOException e) {
            showIOError(e);
        }
    }

    private void clearRoiManager() {
        Roi[] rois = RoiManager.getRoiManager().getRoisAsArray();
        if (rois.length > 0)
            RoiManager.getRoiManager().runCommand("Delete");
    }

    public void submitLabel() {
        if (RoiManager.getRoiManager().getCount() > 0) {
            String imageId = contentPane.getSelectedImageId();
            RoiManager.getRoiManager().runCommand("Save", labelFile.getAbsolutePath());
            try {
                monaiClient.putDatastoreLabel(imageId, labelFile.getAbsolutePath());
            } catch (IOException e) {
                showIOError(e);
                return;
            }
            contentPane.setEnabledLabel(true);
            contentPane.setSelectedLabel(true);
            updateSampleList();
        } else {
            JOptionPane.showMessageDialog(contentPane,
                    "Please add at least one ROI using a selection tool",
                    "Empty ROI Manager",
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    public JSONObject getImageInfo(String imageId) throws IOException {
        JSONObject datastore = monaiClient.getDatastore();
        return datastore.getJSONObject("objects").getJSONObject(imageId);
    }

    public void startTraining() {
        numberOfChannels = getNumberOfChannels();
        JSONObject params = contentPane.getTrainParams();
        validateTrainingParams(params);
        String model = "impartial_" + numberOfChannels;

        try {
            monaiClient.deleteTrain();
            monaiClient.postTrain(model, params);
        } catch (IOException e) {
            showIOError(e);
        }

        monitorTraining();
    }

    private void validateTrainingParams(JSONObject params) {
        if (params.getInt("max_epochs") < 1) {
            JOptionPane.showMessageDialog(contentPane,
                    "Maximum epochs must be greater than 0",
                    "Invalid training parameters",
                    JOptionPane.ERROR_MESSAGE
            );
        }
        if (params.getInt("npatches_epoch") < 1) {
            JOptionPane.showMessageDialog(contentPane,
                    "Number of patches per epoch must be greater than 0",
                    "Invalid training parameters",
                    JOptionPane.ERROR_MESSAGE
            );
        }
        if (params.getInt("early_stop_patience") < 1) {
            JOptionPane.showMessageDialog(contentPane,
                    "Early stop patience must be greater than 0",
                    "Invalid training parameters",
                    JOptionPane.ERROR_MESSAGE
            );
        }
    }

    private void monitorTraining() {
        trainWorker = new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() throws IOException, InterruptedException {
                int lastEpoch = 0;
                int maxEpochs = getMaxEpochs();

                showStatus(lastEpoch, maxEpochs, "Initializing...");

                while (lastEpoch < maxEpochs) {
                    Thread.sleep(1000);
                    monaiClient.getTrain(true);

                    JSONObject jsonProgress = monaiClient.getTrain(false);

                    JSONArray jsonDetails = jsonProgress.getJSONArray("details");
                    List<String> details = new ArrayList<>();
                    for (int i = 0; i < jsonDetails.length(); i++) {
                        details.add(jsonDetails.getString(i));
                    }

                    String[] epochKeywords = {"Epoch:", "train_loss:"};
                    lastEpoch = details.stream()
                            .filter(r -> containsWords(r, epochKeywords))
                            .map(ImpartialController::epochFromLog)
                            .reduce((first, second) -> second)
                            .orElse(0);

                    if (lastEpoch > 0)
                        showStatus(lastEpoch, maxEpochs, "Epoch: " + lastEpoch);
                }
                return null;
            }

            @Override
            protected void done() {
                int maxEpochs = getMaxEpochs();
                contentPane.onTrainingStopped();
                try {
                    get();
                    showStatus(maxEpochs, maxEpochs, "Training done after " + maxEpochs + " epochs");
                } catch (ExecutionException e) {
                    showStatus(0, maxEpochs, "Stopped");
//                    printLogs();
                    JOptionPane.showMessageDialog(contentPane,
                            "An error occurred while training the model. Please check the logs for more information.",
                            "Training stopped",
                            JOptionPane.ERROR_MESSAGE
                    );
                } catch (CancellationException ignore) {
                    showStatus(0, maxEpochs, "Stopped");
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

            private void printLogs() {
                try {
                    System.out.println(monaiClient.getLogs());
                } catch (IOException ignore) {
                }
            }
        };

        trainWorker.execute();

    }

    public void stopTraining() {
        try {
            trainWorker.cancel(true);
            monaiClient.deleteTrain();
        } catch (IOException e) {
            showIOError(e);
        }
    }

    public void infer() {
        ListModel listModel = contentPane.getListModel();

        for (int i = 0; i < listModel.size(); i++) {
            inferImage(listModel.getElementAt(i));
        }
    }

    private void inferImage(Sample sample) {
        SwingWorker<JSONObject, Void> swingWorker = new SwingWorker<JSONObject, Void>() {
            @Override
            protected JSONObject doInBackground() throws IOException {
                contentPane.setSampleStatus(sample, "running");

                String imageId = sample.getName();

                JSONObject params = new JSONObject();
                params.put("threshold", (Float) contentPane.getThreshold());

                String model = "impartial_" + numberOfChannels;
                JSONObject modelOutput = monaiClient.postInferJson(model, imageId, params);

                modelOutput.put("epoch", currentEpoch);

                DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm");
                LocalDateTime time = LocalDateTime.now();
                modelOutput.put("time", time.format(formatter));

                return modelOutput;
            }

            @Override
            protected void done() {
                try {
                    JSONObject modelOutput = get();

                    String imageId = sample.getName();

                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm");
                    LocalDateTime time = LocalDateTime.now();

                    FloatProcessor output = jsonArrayToProcessor(modelOutput.getJSONArray("output"));
                    FloatProcessor entropy = jsonArrayToProcessor(modelOutput.getJSONArray("entropy"));
                    ContrastEnhancer ce = new ContrastEnhancer();
                    ce.equalize(entropy);
                    ce.stretchHistogram(entropy, 0);
                    entropy.setColorModel(redGreenLut);

                    ModelOutput out = new ModelOutput(output, entropy, time.format(formatter), currentEpoch);

                    modelOutputs.put(imageId, out);

                    String selectedImageId = contentPane.getSelectedImageId();
                    if (selectedImageId != null && selectedImageId.equals(imageId)) {
                        updateImage();
                    }

                    contentPane.setSampleStatus(sample, "done");
                    contentPane.setSampleEntropy(sample, entropy.getStatistics().mean);
                    contentPane.sortList();

                } catch (InterruptedException | ExecutionException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        swingWorker.execute();
    }

    public void displayInfer() {
        String imageId = contentPane.getSelectedImageId();
        FloatProcessor processor = modelOutputs.get(imageId).getOutput();
        ImagePlus imp = new ImagePlus("output", processor);

        Img<FloatType> img = ImageJFunctions.wrapFloat(imp);

        FloatType threshold = new FloatType(contentPane.getThreshold());
        Img<BitType> binaryImg = Thresholder.threshold(img, threshold, true, 2);

        final long[] dims = new long[binaryImg.numDimensions()];
        binaryImg.dimensions(dims);

        final RandomAccessibleInterval<UnsignedShortType> indexImg = ArrayImgs.unsignedShorts(dims);
        final ImgLabeling<Integer, UnsignedShortType> labeling = new ImgLabeling<>(indexImg);
        ops.labeling().cca(labeling, binaryImg, ConnectedComponents.StructuringElement.FOUR_CONNECTED);

        LabelRegions<Integer> regions = new LabelRegions<>(labeling);

        for (LabelRegion<Integer> region : regions) {
            Polygon2D contour = regionToPolygonConverter.convert(region, Polygon2D.class);

            int[] xs = contour.vertices().stream().mapToInt(p -> (int) p.getDoublePosition(0)).toArray();
            int[] ys = contour.vertices().stream().mapToInt(p -> (int) p.getDoublePosition(1)).toArray();

            PolygonRoi poly = new PolygonRoi(xs, ys, contour.numVertices(), Roi.POLYGON);

            RoiManager.getRoiManager().add(poly, region.getLabel());
        }
        RoiManager.getRoiManager().runCommand("Show All");
    }

    public void displayEntropy() {
        String imageId = contentPane.getSelectedImageId();
        FloatProcessor entropy = modelOutputs.get(imageId).getEntropy();

        ImageRoi roi = new ImageRoi(0, 0, entropy);
        roi.setZeroTransparent(true);

        Overlay overlay = new Overlay();
        overlay.add(roi);

        ImagePlus image = imageWindow.getImagePlus();
        image.setOverlay(overlay);
    }

    public FloatProcessor jsonArrayToProcessor(JSONArray input) {
        int width = input.length();
        int height = input.getJSONArray(0).length();

        FloatProcessor processor = new FloatProcessor(width, height);

        for (int i = 0; i < input.length(); i++) {
            JSONArray row = input.getJSONArray(i);
            for (int j = 0; j < row.length(); j++) {
                processor.setf(j, i, (float) row.getDouble(j) * 255);
            }
        }

        return processor;
    }

    public void showStatus(int progress, int max, String message) {
        status.showStatus(progress, max, message);
    }

    public int getMaxEpochs() {
        return contentPane.getTrainParams().getInt("max_epochs");
    }

    public void updateDisplay() {
        try {
            List<String> selected = contentPane.getSelectedViews();

            clearRoiManager();
            ImagePlus image = imageWindow.getImagePlus();
            image.setHideOverlay(true);

            if (selected.contains("Entropy")) displayEntropy();
            if (selected.contains("Label")) displayLabel();
            if (selected.contains("Infer")) displayInfer();

            String imageId = contentPane.getSelectedImageId();
            boolean hasLabel = getImageInfo(imageId).getJSONObject("labels").length() > 0;
            contentPane.setEnabledSubmit(
                    !selected.contains("Infer") && (!hasLabel || selected.contains("Label"))
            );

            resetLayout();

        } catch (IOException e) {
            showIOError(e);
        }
    }

    private void resetLayout() {
        Point mainFrameLocation = mainFrame.getLocation();
        imageWindow.setLocation(
                mainFrameLocation.x + mainFrame.getSize().width,
                mainFrameLocation.y
        );

        Point imageWindowLocation = imageWindow.getLocation();
        RoiManager.getRoiManager().setLocation(
                imageWindowLocation.x + imageWindow.getSize().width,
                imageWindowLocation.y
        );
    }

    public void uploadImage(File image) {
        try {
            String filePath = image.getAbsolutePath();
            Opener opener = new Opener();
            ImagePlus imp = opener.openImage(filePath);

            if (numberOfChannels > 0 && getNumberOfChannels(imp) != numberOfChannels) {
                JOptionPane.showMessageDialog(contentPane,
                        "Image " + image.getName() + " has " + imp.getProcessor().getNChannels() + " channels, but the model expects " + numberOfChannels + " channels.",
                        "Upload error",
                        JOptionPane.ERROR_MESSAGE
                );
                return;
            }

            numberOfChannels = getNumberOfChannels(imp);
            monaiClient.putDatastoreImage(image);
        } catch (IOException e) {
            showIOError(e);
        }
    }

    public void uploadImages() {
        int res = fileChooser.showOpenDialog(mainFrame);
        if (res == JFileChooser.APPROVE_OPTION) {
            imageUploader.upload(fileChooser.getSelectedFiles());
        }
    }

    public void deleteSelectedImage() {
        String imageId = contentPane.getSelectedImageId();

        int option = JOptionPane.showConfirmDialog(
                contentPane,
                String.format("Are you sure you want to delete '%s' and corresponding labels?", imageId),
                "Delete confirmation",
                JOptionPane.YES_NO_OPTION
        );

        if (option == JOptionPane.YES_OPTION) {
            try {
                monaiClient.deleteDatastore(imageId);
            } catch (IOException e) {
                showIOError(e);
            }

            numberOfChannels = getNumberOfChannels();
            updateSampleList();
            contentPane.setSelectedFirstImage();
        }
    }

    public void updateSampleList() {
        contentPane.populateSampleList(getImages());
    }

    public JSONObject getImages() {
        return getDatastore().getJSONObject("objects");
    }

    public void startSession() throws IOException {
        try {
            sessionId = sessionClient.postSession();
            scheduleEndOfSessionWarning(110);
        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }

    public JSONObject getSessionDetails() throws IOException {
        try {
            return sessionClient.getSessionDetails();
        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }

    public void downloadModelCheckpoint() {
        String model = "impartial_" + numberOfChannels;
        fileChooser.setSelectedFile(new File(model + ".pt"));
        int res = fileChooser.showSaveDialog(mainFrame);

        if (res == JFileChooser.APPROVE_OPTION) {
            try {
                FileOutputStream outputStream = new FileOutputStream(fileChooser.getSelectedFile());
                outputStream.write(monaiClient.getModel(model));
                outputStream.close();
            } catch (IOException e) {
                showIOError(e);
            }
        }
    }

    public JSONObject getSessions() {
        try {
            return sessionClient.getSessions();
        } catch (IOException e) {
            showIOError(e);
            return null;
        }
    }

    public JFrame getFrame() {
        return mainFrame;
    }

    public void setSession(UserSession selectedSession) {
        if (!sessionId.equals(selectedSession.getId())) {
            sessionId = selectedSession.getId();
            restoreSessionTask.run();
            contentPane.setSession(selectedSession.getId());
        }
    }

    private void scheduleEndOfSessionWarning(int delayInMinutes) {
        if (endOfSessionWarningTask != null)
            endOfSessionWarningTask.cancel();

        endOfSessionWarningTask = new TimerTask() {
            @Override
            public void run() {
                int choice = JOptionPane.showOptionDialog(null,
                        "Your session is about to expire.", "Session expiration warning",
                        JOptionPane.YES_NO_OPTION, JOptionPane.QUESTION_MESSAGE,
                        null, new String[]{"Extend", "Ok"}, "Ok");

                if (choice == JOptionPane.YES_OPTION) {
                    try {
                        sessionClient.extendSession();
                        scheduleEndOfSessionWarning(25);
                    } catch (IOException e) {
                        showIOError(e);
                    }
                }
            }
        };

        timer.schedule(endOfSessionWarningTask, 1000L * 60 * delayInMinutes);
    }

    public String getSessionId() {
        return sessionId;
    }

    public void login(String username, String password) throws IOException {
        try {
            String token = sessionClient.postLogin(username, password);
            sessionClient.setToken(token);
            monaiClient = new ProxyMonaiLabelClient(Config.API_URL, token);
        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }
}
