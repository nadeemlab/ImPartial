package org.nadeemlab.impartial;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;

public class ServerPanel extends JPanel {
    private final JLabel statusLabel;
    private final JTextField monaiUrlTextField;
    private final JCheckBox requestServerCheckBox;
    private final JButton startButton;
    private final JButton stopButton;
    private String url = "http://localhost:8000";
    private boolean warningDisplayed = false;

    ServerPanel(ImpartialController controller) {

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("server"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

        monaiUrlTextField = new JTextField(url, 16);
        monaiUrlTextField.setAlignmentX(LEFT_ALIGNMENT);

        startButton = new JButton("start");
        startButton.setActionCommand("start");

        stopButton = new JButton("stop");
        stopButton.setEnabled(false);
        stopButton.setActionCommand("stop");

        String terms = "PLEASE READ THIS DOCUMENT CAREFULLY BEFORE YOU ACCESS OR USE IMPARTIAL. BY ACCESSING ANY PORTION OF IMPARTIAL, YOU AGREE TO BE BOUND BY THE TERMS AND CONDITIONS SET FORTH BELOW. IF YOU DO NOT WISH TO BE BOUND BY THESE TERMS AND CONDITIONS, PLEASE DO NOT ACCESS IMPARTIAL.\n" +
                "\n" +
                "ImPartial is developed and maintained by Memorial Sloan Kettering Cancer Center (“MSK,” “we”, or “us”) to provide an experimental artificial intelligence driven tool to perform segmentation using as few as 2-3 training images with some user-provided scribbles. MSK may, from time to time, update the software and other content on https://impartial.mskcc.org/ (“Content”). MSK makes no warranties or representations, and hereby disclaims any warranties, express or implied, with respect to any of the Content, including as to the present accuracy, completeness, timeliness, adequacy, or usefulness of any of the Content. By using this service, you agree that MSK will not be liable for any losses or damages arising from your use of or reliance on the Content, or other websites or information to which this website may be linked. \n" +
                "\n" +
                "By providing images to be analyzed using ImPartial (Submitted Images), you authorize MSK to copy, modify, display, distribute, perform, use, publish, and otherwise exploit the Submitted Images for any and all purposes, all without compensation to you, for as long as we decide (collectively, the \"Use Rights\"). In addition, you authorize us to grant any third party some or all of the Use Rights. By way of example, and not limitation, the Use Rights include the right for us to publish Submitted Images in whole or in part, and whether cropped, adopted, altered, or otherwise manipulated, for as long as we choose. By providing Submitted Images, you represent and warrant that (i) you own all rights in and to the Submitted Images (including any related copyrights or other intellectual property rights) or have sufficient authority and right to provide the content and to grant the Use Rights; (ii) your submission of the images and grant to us of Use Rights do not violate or conflict with the rights of other persons, or breach your obligations to other persons; and (iii) the Submitted Images do not include or contain any personally identifiable information (PII) or protected health information (PHI).\n" +
                "\n" +
                "DO NOT submit personally identifiable information (PII) or protected health information (PHI) in connection with any Submitted Images or otherwise.  \n" +
                "\n" +
                "You may use ImPartial the underlying Content, and images output therefrom for personal or academic research purposes only.  You may not use it for any other purpose, including not but limited to use for diagnosis, treatment, or patient care.  You may publish in scientific or academic journals or literature the results of such research, subject to providing credit to MSK as the source of output of ImPartial and with reference to these Terms of Use.  You may not otherwise redistribute or share the Content with any third party, in part or in whole, for any purpose, without the express written permission of MSK.\n" +
                "\n" +
                "Without limiting the generality of the foregoing, you may not use any part of the ImPartial, the underlying Content or the output for any other purpose, including:\n" +
                "\n" +
                "(i) use or incorporation into a commercial product or towards the performance of a commercial service;\n" +
                "(ii) research use in a commercial setting;\n" +
                "(ii) use for patient care or the provision of medical services; or\n" +
                "(iv) generation of reports in a medical, laboratory, hospital or other patient care setting.\n" +
                "\n" +
                "\n" +
                "You may not copy, transfer, reproduce, modify or create derivative works of ImPartial or the underlying Content for any commercial purpose without the express permission of MSK. \n" +
                "\n" +
                "The model output of ImPartial and the underlying Content is not a substitute for professional medical help, judgment or advice. Use of ImPartial does not create a physician-patient relationship or in any way make a person a patient of MSK. A physician or other qualified health provider should always be consulted for any health problem or medical condition. \n" +
                "\n" +
                "Neither these Terms or Use nor the availability of ImPartial should be understood to create an obligation or expectation that MSK will continue to make ImPartial available. MSK may discontinue or restrict the availability of ImPartial at any time. MSK may also modify these Terms or Use at any time.\n" +
                "\n" +
                "MSK respects the intellectual property rights of others, just as it expects others to respect its intellectual property. If you believe that any content (including Submitted Images and Content) on the website or other activity taking place on the website constitutes infringement of a work protected by copyright, please notify us at:\n" +
                "\n" +
                "cmsdigitalteam@mskcc.org\n" +
                "\n" +
                "Your notice must comply with the Digital Millennium Copyright Act (17 U.S.C. §512) (the \"DMCA\"). Upon receipt of a compliant notice, we will respond and proceed in accordance with the DMCA.\n" +
                "\n" +
                "By using ImPartial, you consent to the jurisdiction and venue of the state and federal courts located in New York City, New York, USA, for any claims related to or arising from your use of ImPartial or your violation of these Terms of Use and agree that you will not bring any claims against MSK that relate to or arise from the foregoing except in those courts. \n" +
                "\n" +
                "If any provision of these Terms of Use is held to be invalid or unenforceable, then such provision shall be struck, and the remaining provisions shall be enforced. Headings are for reference purposes only and in no way define, limit, construe, or describe the scope or extent of such section. MSK’s failure to act with respect to a breach by you or others does not waive its right to act with respect to subsequent or similar breaches. This agreement and the terms and conditions contained herein set forth the entire understanding and agreement between MSK and you with respect to the subject matter hereof and supersede any prior or contemporaneous understanding, whether written or oral.\n" +
                "\n" +
                "For inquiries about the Content, please contact us at nadeems@mskcc.org.\n" +
                "\n" +
                "If you are interested in using ImPartial for purposes beyond those permitted by these Terms of Use, please contact us at nadeems@mskcc.org to inquire concerning the availability of a license.\n" +
                "" ;

        requestServerCheckBox = new JCheckBox("request server");
        requestServerCheckBox.addActionListener(e -> {
            if (!warningDisplayed) {
                warningDisplayed = true;
                JTextArea textArea = new JTextArea(terms);
                JScrollPane scrollPane = new JScrollPane(textArea);
                textArea.setLineWrap(true);
                textArea.setWrapStyleWord(true);
                scrollPane.setPreferredSize( new Dimension( 500, 500 ) );
                JOptionPane.showMessageDialog(null, scrollPane, "ImPartial Terms of Use",
                        JOptionPane.WARNING_MESSAGE);
            }

            startButton.setEnabled(true);

            if (requestServerCheckBox.isSelected()) {
                monaiUrlTextField.setEnabled(false);
                monaiUrlTextField.setText("https://impartial.mskcc.org");
            } else {
                monaiUrlTextField.setEnabled(true);
                monaiUrlTextField.setText(url);
            }
        });

        startButton.addActionListener(e -> {
            startButton.setEnabled(false);
            if (!requestServerCheckBox.isSelected())
                url = monaiUrlTextField.getText();

            controller.connect();
        });

        stopButton.addActionListener(e -> {
            startButton.setEnabled(false);
            controller.disconnect();
        });

        monaiUrlTextField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startButton.setEnabled(true);
            }
        });

        JPanel innerPanel = new JPanel();
        innerPanel.setLayout(new BoxLayout(innerPanel, BoxLayout.LINE_AXIS));
        innerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JPanel lowInnerPanel = new JPanel();
        lowInnerPanel.setLayout(new BoxLayout(lowInnerPanel, BoxLayout.LINE_AXIS));
        lowInnerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        statusLabel = new JLabel();
        statusLabel.setText("status: disconnected");

        innerPanel.add(startButton);
        innerPanel.add(requestServerCheckBox);
        innerPanel.add(Box.createHorizontalGlue());

        lowInnerPanel.add(stopButton);
        lowInnerPanel.add(statusLabel);
        lowInnerPanel.add(Box.createHorizontalGlue());

        content.add(monaiUrlTextField);
        content.add(innerPanel);
        content.add(lowInnerPanel);

        add(content);
    }

    public URL getUrl() throws MalformedURLException {
        return new URL(url);
    }

    public void setStatusLabel(String status) {
        statusLabel.setText(status);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onConnected() {
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        requestServerCheckBox.setEnabled(false);
        statusLabel.setText("status: connected");
    }

    public void onDisconnected() {
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        requestServerCheckBox.setEnabled(true);
        statusLabel.setText("status: disconnected");
    }
}
