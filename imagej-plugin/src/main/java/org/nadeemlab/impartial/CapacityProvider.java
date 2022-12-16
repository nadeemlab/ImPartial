package org.nadeemlab.impartial;

import java.awt.*;
import javax.swing.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;

public class CapacityProvider implements PropertyChangeListener {
    private final ImpartialController controller;
    private Task task;
    private ProgressMonitor progressMonitor;

    public CapacityProvider(ImpartialController controller) {
        this.controller = controller;
    }

    public void provisionServer() {
        progressMonitor = new ProgressMonitor(
                controller.getContentPane(), "request server...", "", 0, 100
        );

        progressMonitor.setProgress(0);
        progressMonitor.setMillisToDecideToPopup(0);
        progressMonitor.setMillisToPopup(0);

        task = new Task();
        task.addPropertyChangeListener(this);
        task.execute();
    }

    class Task extends SwingWorker<Void, Void> {
        @Override
        public Void doInBackground() {
            setProgress(20);
            try {
                String token = controller.startSession();
                String lastStatus = controller.getSessionStatus(token).getString("last_status");

                while (!lastStatus.equals("RUNNING") && !isCancelled()) {
                    Thread.sleep(1000);
                    lastStatus = controller.getSessionStatus(token).getString("last_status");

                    if (lastStatus.equals("PROVISIONING")) setProgress(40);
                    else if (lastStatus.equals("PENDING")) setProgress(60);
                }

                setProgress(80);
                String healthStatus = controller.getSessionStatus(token).getString("health_status");
                while (!healthStatus.equals("HEALTHY") && !isCancelled()) {
                    Thread.sleep(1000);
                    healthStatus = controller.getSessionStatus(token).getString("health_status");
                }

                setProgress(100);

            } catch (InterruptedException ignore) {}
            return null;
        }

        @Override
        public void done() {
            Toolkit.getDefaultToolkit().beep();
            controller.onConnected();
        }
    }

    /**
     * Invoked when task's progress property changes.
     */
    public void propertyChange(PropertyChangeEvent e) {
        if (e.getPropertyName().equals("progress")) {
            int progress = (Integer) e.getNewValue();
            progressMonitor.setProgress(progress);
            String message;

            if (progress == 20) message = "validating token";
            else if (progress == 40) message = "provisioning server";
            else if (progress == 60) message = "task pending";
            else message = "starting server";
//            else if (progress == 80) message = "starting server";

            progressMonitor.setNote(message);

            if (progressMonitor.isCanceled() || task.isDone()) {
                Toolkit.getDefaultToolkit().beep();
                if (progressMonitor.isCanceled()) {
                    task.cancel(true);
                } else {
                    progressMonitor.close();
                }
            }
        }
    }
}