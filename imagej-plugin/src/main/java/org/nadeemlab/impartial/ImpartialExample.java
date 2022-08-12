package org.nadeemlab.impartial;

import javax.swing.*;
import net.imagej.ImageJ;

public class ImpartialExample {

	private void createAndShowGUI(ImageJ ij) {
		//Create and set up the window.
		JFrame frame = new JFrame("ImPartial");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		//Create and set up the content pane.
		ImpartialDialog impartial = new ImpartialDialog(ij.context());
		impartial.mainPane.setOpaque(true); //content panes must be opaque
		frame.setContentPane(impartial.mainPane);

		//Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	public static void main(final String[] args) {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		try {
			//Create and set up the window.
			JFrame frame = new JFrame("ImPartial");
			frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			//Create and set up the content pane.
			ImpartialDialog impartial = new ImpartialDialog(ij.context());
			impartial.mainPane.setOpaque(true); //content panes must be opaque
			frame.setContentPane(impartial.mainPane);

			//Display the window.
			frame.pack();
			frame.setVisible(true);
		}
		catch (final Exception e) {
			e.printStackTrace();
		}
	}

}
