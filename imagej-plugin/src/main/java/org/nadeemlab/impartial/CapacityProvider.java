package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.IOException;

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

    /**
     * Invoked when task's progress property changes.
     */
    public void propertyChange(PropertyChangeEvent e) {
        if (e.getPropertyName().equals("progress")) {
            int progress = (Integer) e.getNewValue();
            progressMonitor.setProgress(progress);

            if (progressMonitor.isCanceled() || task.isDone()) {
                if (progressMonitor.isCanceled()) {
                    task.cancel(true);
                } else {
                    progressMonitor.close();
                }
            }
        }
    }

    class Task extends SwingWorker<Void, Void> {
        @Override
        public Void doInBackground() {
            progressMonitor.setNote("validating token");
            setProgress(20);

            try {
                controller.startSession();

                String lastStatus = controller.getSessionStatus().getString("last_status");

                while (!lastStatus.equals("RUNNING") && !isCancelled() && !progressMonitor.isCanceled()) {
                    Thread.sleep(1000);
                    lastStatus = controller.getSessionStatus().getString("last_status");

                    if (lastStatus.equals("PROVISIONING")) {
                        progressMonitor.setNote("provisioning server");
                        setProgress(40);
                    }
                    else if (lastStatus.equals("PENDING")) {
                        progressMonitor.setNote("task pending");
                        setProgress(60);
                    }

                }

                progressMonitor.setNote("starting server");
                setProgress(80);

                String healthStatus = controller.getSessionStatus().getString("health_status");
                while (!healthStatus.equals("HEALTHY") && !isCancelled() && !progressMonitor.isCanceled()) {
                    Thread.sleep(1000);
                    healthStatus = controller.getSessionStatus().getString("health_status");
                }

                progressMonitor.setNote("done");
                setProgress(100);

            } catch (IOException e) {
                this.cancel(true);
                progressMonitor.close();

                return null;
            } catch (InterruptedException ignore) {}

            return null;
        }

        @Override
        public void done() {
            Toolkit.getDefaultToolkit().beep();
            if (isCancelled()) controller.disconnect();
            else controller.onConnected();
        }
    }
}