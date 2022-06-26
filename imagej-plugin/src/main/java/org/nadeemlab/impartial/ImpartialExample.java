package org.nadeemlab.impartial;

import javax.swing.*;
import net.imagej.ImageJ;

public class ImpartialExample {

	public static void main(final String[] args) {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		try {
			final ImpartialDialog dialog = new ImpartialDialog(ij.context());
			dialog.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
			dialog.setVisible(true);
		}
		catch (final Exception e) {
			e.printStackTrace();
		}
	}

}
