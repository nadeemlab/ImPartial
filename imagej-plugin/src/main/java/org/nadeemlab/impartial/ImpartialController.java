package org.nadeemlab.impartial;

import ij.ImagePlus;
import ij.gui.*;
import ij.io.Opener;
import ij.plugin.ContrastEnhancer;
import ij.plugin.LutLoader;
import ij.plugin.frame.RoiManager;
import ij.process.ColorProcessor;
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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.StreamSupport;


public class ImpartialController {
    final JFileChooser fileChooser = new JFileChooser();
    private final JFrame mainFrame;
    private final ImpartialContentPane contentPane;
    private final MonaiLabelClient monaiClient = new MonaiLabelClient();
    private final ImpartialClient impartialClient = new ImpartialClient();
    private final Hashtable<String, ModelOutput> modelOutputs = new Hashtable<>();
    private final CapacityProvider capacityProvider;
    private final ImageUploader imageUploader;
    private final IndexColorModel redGreenLut;
    LabelRegionToPolygonConverter regionToPolygonConverter = new LabelRegionToPolygonConverter();
    private ImageWindow imageWindow;
    private File imageFile;
    private File labelFile;
    private int currentEpoch = 0;
    @Parameter
    private OpService ops;
    @Parameter
    private StatusService status;

    public ImpartialController(final Context context) {
        context.inject(this);
        context.inject(regionToPolygonConverter);
        createTempFiles();

        fileChooser.setMultiSelectionEnabled(true);

        mainFrame = new JFrame("ImPartial");
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        mainFrame.setResizable(false);

        //Create and set up the content pane.
        contentPane = new ImpartialContentPane(this);
        contentPane.setOpaque(true); //content panes must be opaque
        mainFrame.setContentPane(contentPane);

        mainFrame.pack();
        mainFrame.setVisible(true);

        capacityProvider = new CapacityProvider(this);
        imageUploader = new ImageUploader(this);

        redGreenLut = LutLoader.getLut("redgreen");
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

    public void connect() {
        if (contentPane.getRequestServerCheckBox()) {
            try {
                monaiClient.setUrl(
                        new URL(String.format("https://%s:%s",
                                impartialClient.getHost(),
                                impartialClient.getPort()
                        ))
                );
            } catch (MalformedURLException ignore) {
            }

            capacityProvider.provisionServer();

        } else {
            try {
                monaiClient.setUrl(contentPane.getUrl());
                monaiClient.getInfo();
            } catch (IOException e) {
                onDisconnected();
                showIOError(e);
                return;
            }
            onConnected();
        }
    }

    public void disconnect() {
        if (contentPane.getRequestServerCheckBox()) {
            try {
                impartialClient.stopSession();
            } catch (IOException e) {
                showIOError(e);
            }
        }
        onDisconnected();
    }

    public void onConnected() {
        updateSampleList();
        contentPane.onConnected();
    }

    public void onDisconnected() {
        contentPane.onDisconnected();
        modelOutputs.clear();
        monaiClient.setToken(null);
        impartialClient.setToken(null);

        if (imageWindow != null) {
            imageWindow.setVisible(false);
        }
    }

    public void updateImage() {
        try {
            clearRoiManager();
            String imageId = contentPane.getSelectedImageId();

            displayImage();

            JSONObject imageInfo = getImageInfo(imageId);
            boolean hasLabel = imageInfo.getJSONObject("labels").length() > 0;

            contentPane.setEnabledLabel(hasLabel);
            contentPane.setEnabledSubmit(hasLabel && contentPane.getSelectedViews().contains("label"));
            if (!hasLabel)
                contentPane.setSelectedLabel(false);

            contentPane.setEnabledInfer(true);
            if (modelOutputs.containsKey(imageId)) {
                ModelOutput output = modelOutputs.get(imageId);
                contentPane.setTextInfer(
                        "last run " + output.getTime() + ", epoch " + output.getEpoch()
                );

                contentPane.setEnabledInferAndEntropy(true);
            } else {
                contentPane.setTextInfer("last run never");
                contentPane.setEnabledInferAndEntropy(false);
            }

            updateDisplay();

        } catch (IOException e) {
            showIOError(e);
        }
    }

    public void displayImage() {
        String imageId = contentPane.getSelectedImageId();
        try {
            byte[] imageBytes = monaiClient.getDatastoreImage(imageId);

            FileOutputStream stream = new FileOutputStream(imageFile.getAbsolutePath());
            stream.write(imageBytes);
            stream.close();
        } catch (IOException e) {
            showIOError(e);
        }

        final ImagePlus imp = new Opener().openImage(imageFile.getAbsolutePath());
        displayImage(imp);
    }

    private void displayImage(ImagePlus imp) {
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

    public String[] getDatastoreSamples() {
        try {
            JSONObject datastore = monaiClient.getDatastore();

            Iterable<String> iterable = () -> datastore.getJSONObject("objects").keys();
            return StreamSupport.stream(iterable.spliterator(), false)
                    .toArray(String[]::new);
        } catch (IOException e) {
            showIOError(e);
            return new String[0];
        }
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

    public void train() {
        JSONObject params = contentPane.getTrainParams();
        String model = "impartial_" + params.getInt("num_channels");

        try {
            monaiClient.deleteTrain();
            monaiClient.postTrain(model, params);
        } catch (IOException e) {
            showIOError(e);
        }

        monitorTraining();
    }

    private void monitorTraining() {
        SwingWorker<Void, Void> swingWorker = new SwingWorker<Void, Void>() {
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

            private void printLogs() {
                try {
                    System.out.println(monaiClient.getLogs());
                } catch (IOException ignore) {
                }

            }

            @Override
            protected void done() {
                int maxEpochs = getMaxEpochs();
                try {
                    get();
                    showStatus(maxEpochs, maxEpochs, "Training done after " + maxEpochs + " epochs");
                } catch (ExecutionException e) {
                    showStatus(0, maxEpochs, "Stopped");
                    printLogs();
                    showIOError((IOException) e.getCause());
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            }
        };

        swingWorker.execute();
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

                String model = "impartial_" + contentPane.getTrainParams().getInt("num_channels");
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

    private void inferPerformed(String imageId) {
        ModelOutput modelOutput = modelOutputs.get(imageId);
        contentPane.inferPerformed(
                modelOutput.getEpoch(),
                modelOutput.getTime()
        );

        displayInfer();
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

    public ColorProcessor jsonArrayToColorProcessor(JSONArray input) {
        int width = input.length();
        int height = input.getJSONArray(0).length();

        ColorProcessor processor = new ColorProcessor(width, height);

        for (int i = 0; i < input.length(); i++) {
            JSONArray row = input.getJSONArray(i);
            for (int j = 0; j < row.length(); j++) {
                processor.putPixel(j, i, (int) (row.getDouble(j) * 255));
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

            if (selected.contains("entropy")) displayEntropy();
            if (selected.contains("label")) displayLabel();
            if (selected.contains("infer")) displayInfer();

            String imageId = contentPane.getSelectedImageId();
            boolean hasLabel = getImageInfo(imageId).getJSONObject("labels").length() > 0;
            contentPane.setEnabledSubmit(
                    !selected.contains("infer") && (!hasLabel || selected.contains("label"))
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
            monaiClient.putDatastore(image);
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

    public void updateSampleList() {
        String[] samples = getDatastoreSamples();
        contentPane.populateSampleList(samples);
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

            updateSampleList();

            contentPane.setSelectedFirstImage();
        }
    }

    public void startSession() throws IOException {
        try {
            String token = impartialClient.createSession().getString("token");
            impartialClient.setToken(token);
            monaiClient.setToken(token);

        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }

    public JSONObject getSessionStatus() throws IOException {
        try {
            return impartialClient.sessionStatus();
        } catch (IOException e) {
            showIOError(e);
            throw e;
        }
    }

    public void downloadModelCheckpoint() {
        String model = "impartial_" + contentPane.getTrainParams().getInt("num_channels");
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
}
